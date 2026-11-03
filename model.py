"""Mixture of Smaller Experts (MoSE) Architecture (model.py).

Implements the MoSE Causal Language Model designed for fine-grained Python code
generation and Fill-in-the-Middle infilling under 2.5 GB peak VRAM.

Components:
- MicroExpert: Ultra-compact FFN (Linear -> GELU -> Linear)
- MoSEBlock: Gating router, top-k dispatch, auxiliary load-balancing loss, shared expert
- TransformerBlock: Pre-LayerNorm, Causal Multi-Head Self-Attention, MoSE Feed-Forward
- MoSELanguageModel: Embeddings, Transformer layers, LayerNorm, LM head, generation
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ==============================================================================
# 1. Micro-Expert Module
# ==============================================================================

class MicroExpert(nn.Module):
    """Ultra-compact Feed-Forward Expert unit.
    
    Structure: Linear(d_model -> expert_dim) -> GELU -> Linear(expert_dim -> d_model)
    """

    def __init__(self, d_model: int, expert_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expert_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expert_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ==============================================================================
# 2. Causal Multi-Head Self-Attention
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-Head Attention with causal autoregressive masking."""

    def __init__(self, config: Config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Causal mask buffer
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
            1, 1, config.seq_len, config.seq_len
        )
        self.register_buffer("bias", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


# ==============================================================================
# 3. MoSE Block: Gating Router, Micro-Experts & Shared Expert
# ==============================================================================

class MoSEBlock(nn.Module):
    """Mixture of Smaller Experts (MoSE) Feed-Forward Layer.
    
    Features:
    - Gating Network: Routes tokens to top-k micro-experts
    - Shared Expert: Persistent baseline expert processing every token
    - Auxiliary Load Balancing Loss: Prevents expert collapse via Switch Transformer loss
    - Sparse Dispatch & Gather: Computes only active micro-experts
    """

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_model = config.d_model
        self.has_shared_expert = config.has_shared_expert

        # 1. Gating router
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)

        # 2. Micro-Expert Swarm
        self.experts = nn.ModuleList([
            MicroExpert(config.d_model, config.expert_dim)
            for _ in range(config.num_experts)
        ])

        # 3. Shared Base Expert (Optional baseline path)
        if self.has_shared_expert:
            self.shared_expert = MicroExpert(config.d_model, config.shared_expert_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass through MoSE block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_weights: If True, returns top_k_weights, raw confidence scores, and gate_probs
            
        Returns:
            If return_weights is False:
                out: Combined output tensor [batch_size, seq_len, d_model]
                aux_loss: Load balancing auxiliary loss scalar
                top_k_indices: Routing indices [batch_size * seq_len, top_k]
            If return_weights is True:
                out, aux_loss, top_k_indices, top_k_weights, top_k_raw_weights, gate_probs
        """
        B, T, C = x.shape
        flat_x = x.view(-1, C)  # [N_tokens, d_model]
        n_tokens = flat_x.size(0)

        # Compute gating router logits and probabilities
        gate_logits = self.gate(flat_x)  # [N_tokens, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [N_tokens, num_experts]

        # Select Top-k experts
        top_k_raw_weights, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Normalize weights so they sum to 1 over the selected top-k
        top_k_weights = top_k_raw_weights / (top_k_raw_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ----------------------------------------------------------------------
        # Auxiliary Load Balancing Loss: N * sum(f_i * P_i)
        # ----------------------------------------------------------------------
        # P_i: Average probability assigned to expert i
        P_i = gate_probs.mean(dim=0)  # [num_experts]
        
        # f_i: Fraction of tokens where expert i is ranked top-1
        top_1_indices = top_k_indices[:, 0]
        f_one_hot = F.one_hot(top_1_indices, num_classes=self.num_experts).float()
        f_i = f_one_hot.mean(dim=0)  # [num_experts]
        
        aux_loss = self.num_experts * torch.sum(f_i * P_i)

        # ----------------------------------------------------------------------
        # Sparse Dispatch & Gather
        # ----------------------------------------------------------------------
        expert_out = torch.zeros_like(flat_x)

        # Vectorized gather across active expert assignments
        for k_idx in range(self.top_k):
            chosen_experts = top_k_indices[:, k_idx]
            weights = top_k_weights[:, k_idx:k_idx + 1]

            for e_idx in range(self.num_experts):
                mask = (chosen_experts == e_idx)
                if mask.any():
                    tokens_for_expert = flat_x[mask]
                    processed = self.experts[e_idx](tokens_for_expert)
                    expert_out[mask] += weights[mask] * processed

        # ----------------------------------------------------------------------
        # Shared Expert (Unrouted baseline path)
        # ----------------------------------------------------------------------
        if self.has_shared_expert:
            shared_out = self.shared_expert(flat_x)
            out = shared_out + expert_out
        else:
            out = expert_out

        if return_weights:
            return out.view(B, T, C), aux_loss, top_k_indices, top_k_weights, top_k_raw_weights, gate_probs

        return out.view(B, T, C), aux_loss, top_k_indices


# ==============================================================================
# 4. Transformer Block: Pre-LayerNorm Attention + MoSE
# ==============================================================================

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with Causal Attention & MoSE FFN."""

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mose = MoSEBlock(config)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ):
        # Pre-LN Self-Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # Pre-LN MoSE with residual connection
        if return_weights:
            mose_out, aux_loss, routing_indices, top_k_weights, top_k_raw_weights, gate_probs = self.mose(
                self.ln_2(x), return_weights=True
            )
            x = x + mose_out
            return x, aux_loss, routing_indices, top_k_weights, top_k_raw_weights, gate_probs
        else:
            mose_out, aux_loss, routing_indices = self.mose(self.ln_2(x))
            x = x + mose_out
            return x, aux_loss, routing_indices


# ==============================================================================
# 5. Complete MoSE Causal Language Model
# ==============================================================================

class MoSELanguageModel(nn.Module):
    """Full MoSE Causal Language Model for Python Code Generation & FIM."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Token & Positional Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.seq_len, config.d_model)

        # Transformer Layers
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final LayerNorm & LM Head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Embeddings tied to LM Head for parameter efficiency & stability)
        self.lm_head.weight = self.wte.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs
            targets: Optional [batch_size, seq_len] tensor for cross-entropy loss
            
        Returns:
            dict containing:
                - logits: [batch_size, seq_len, vocab_size]
                - total_loss: task_loss + aux_loss_weight * aux_loss
                - task_loss: cross-entropy loss
                - aux_loss: load-balancing loss
                - routing_indices: list of routing decisions per layer
        """
        B, T = input_ids.shape
        assert T <= self.config.seq_len, f"Sequence length {T} exceeds maximum {self.config.seq_len}"

        # Positions tensor: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embeddings: token + positional
        x = self.wte(input_ids) + self.wpe(pos)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        layer_routing = []

        # Pass through Transformer blocks
        for block in self.blocks:
            x, block_aux, routing_indices = block(x)
            total_aux_loss = total_aux_loss + block_aux
            layer_routing.append(routing_indices)

        # Average auxiliary loss across layers
        total_aux_loss = total_aux_loss / self.config.n_layers

        # Final LayerNorm & Language Modeling Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        total_loss = None
        task_loss = None

        if targets is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_targets = targets[..., 1:].contiguous().view(-1)

            task_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)
            total_loss = task_loss + self.config.aux_loss_weight * total_aux_loss

        return {
            "logits": logits,
            "total_loss": total_loss,
            "task_loss": task_loss,
            "aux_loss": total_aux_loss,
            "routing_indices": layer_routing,
        }

    def count_parameters(self) -> Dict[str, any]:
        """Calculates total, unique, and active parameters per token."""
        unique_params = sum(p.numel() for p in set(self.parameters()))
        
        # Calculate active parameters per token:
        # Embedding + Position + Final LN + LN1 + Attention + LN2 + Shared Expert + Top_k * MicroExpert + Gate
        emb_params = self.wte.weight.numel() + self.wpe.weight.numel()
        final_ln_params = sum(p.numel() for p in self.ln_f.parameters())

        layer_active = 0
        b0 = self.blocks[0]
        layer_active += sum(p.numel() for p in b0.ln_1.parameters())
        layer_active += sum(p.numel() for p in b0.attn.parameters())
        layer_active += sum(p.numel() for p in b0.ln_2.parameters())
        layer_active += sum(p.numel() for p in b0.mose.gate.parameters())
        if self.config.has_shared_expert:
            layer_active += sum(p.numel() for p in b0.mose.shared_expert.parameters())
        # Only top_k micro-experts are executed per token
        single_expert_params = sum(p.numel() for p in b0.mose.experts[0].parameters())
        layer_active += single_expert_params * self.config.top_k

        active_params = emb_params + (layer_active * self.config.n_layers) + final_ln_params

        return {
            "total_parameters": unique_params,
            "active_parameters_per_token": active_params,
            "sparsity_ratio": 1.0 - (active_params / unique_params),
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generates continuation tokens given a prompt."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]

            outputs = self.forward(idx_cond)
            logits = outputs["logits"][:, -1, :]  # Next-token logits [B, vocab_size]

            if temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return idx

    @torch.no_grad()
    def inspect_forward(
        self,
        input_ids: torch.Tensor,
        tokenizer: Optional[any] = None,
    ) -> Dict[str, any]:
        """Runs tokens through embedding, attention, and MoSE layers, collecting full routing trace.
        
        For every token t in prompt:
            Trace(t) = { Layer l : [ (Expert A, Weight w_A), (Expert B, Weight w_B) ] }
            
        Args:
            input_ids: [batch_size, seq_len] or [seq_len] tensor of token IDs
            tokenizer: Optional Tokenizer instance to decode token strings
            
        Returns:
            dict containing:
                - logits: [B, T, vocab_size] next-token prediction logits
                - trace: List of length T with Trace(t) dict for each token
                - tokens: List of decoded token strings
                - token_labels: List of display strings safe for UI and heatmaps
                - token_ids: List of integer token IDs
                - layer_routing: Dict indexed by layer_idx (0..n_layers-1) containing:
                      - indices: [T, top_k] numpy array
                      - weights: [T, top_k] normalized weights (sum to 1)
                      - raw_weights: [T, top_k] softmax confidence scores
                      - gate_probs: [T, num_experts] full expert distribution
                      - sparse_matrix: [num_experts, T] matrix of active weights for heatmap
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, T = input_ids.shape
        assert T <= self.config.seq_len, f"Sequence length {T} exceeds maximum {self.config.seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)

        layer_routing = {}

        for l_idx, block in enumerate(self.blocks):
            x, _, indices, weights, raw_weights, gate_probs = block(x, return_weights=True)

            # Reshape from [B*T, ...] to [T, ...]
            ind_np = indices.view(T, self.config.top_k).cpu().numpy()
            wt_np = weights.view(T, self.config.top_k).cpu().numpy()
            raw_wt_np = raw_weights.view(T, self.config.top_k).cpu().numpy()
            probs_np = gate_probs.view(T, self.config.num_experts).cpu().numpy()

            # Construct sparse [num_experts, T] matrix for Plotly heatmap
            sparse_mat = np.zeros((self.config.num_experts, T), dtype=np.float32)
            for t in range(T):
                for k in range(self.config.top_k):
                    exp = ind_np[t, k]
                    sparse_mat[exp, t] = wt_np[t, k]

            layer_routing[l_idx] = {
                "indices": ind_np,
                "weights": wt_np,
                "raw_weights": raw_wt_np,
                "gate_probs": probs_np,
                "sparse_matrix": sparse_mat,
            }

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Build token labels
        token_ids_list = input_ids[0].tolist()
        token_strings = []
        token_labels = []
        if tokenizer is not None:
            for tid in token_ids_list:
                raw_token = tokenizer.id_to_token(tid)
                decoded = tokenizer.decode([tid])
                if raw_token is None:
                    raw_token = decoded
                token_strings.append(decoded)
                
                # Clean readable label for UI & Plotly x-axis
                lbl = raw_token.replace('\u0120', ' ').replace('\u010a', '↵').replace('\u0109', '⇥')
                token_labels.append(lbl)
        else:
            token_strings = [f"id_{tid}" for tid in token_ids_list]
            token_labels = [f"id_{tid}" for tid in token_ids_list]

        # Trace(t) = { Layer l : [(Expert A, Weight w_A), (Expert B, Weight w_B)] }
        trace = []
        for t in range(T):
            t_trace = {}
            for l_idx in range(self.config.n_layers):
                l_data = layer_routing[l_idx]
                t_trace[l_idx] = [
                    (int(l_data["indices"][t, k]), float(l_data["weights"][t, k]))
                    for k in range(self.config.top_k)
                ]
            trace.append(t_trace)

        return {
            "logits": logits,
            "trace": trace,
            "tokens": token_strings,
            "token_labels": token_labels,
            "token_ids": token_ids_list,
            "layer_routing": layer_routing,
        }

    @torch.no_grad()
    def inspect_generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_id: Optional[int] = None,
        tokenizer: Optional[any] = None,
    ) -> Dict[str, any]:
        """Autoregressively generates continuation tokens and logs routing at each generation step."""
        self.eval()
        curr_ids = idx.clone()
        if curr_ids.dim() == 1:
            curr_ids = curr_ids.unsqueeze(0)

        prompt_len = curr_ids.size(1)
        generated_token_ids = []

        for _ in range(max_new_tokens):
            idx_cond = curr_ids if curr_ids.size(1) <= self.config.seq_len else curr_ids[:, -self.config.seq_len:]
            outputs = self.forward(idx_cond)
            logits = outputs["logits"][:, -1, :]

            if temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled_logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat((curr_ids, next_token), dim=1)
            generated_token_ids.append(next_token.item())

            if eos_id is not None and (next_token == eos_id).all():
                break

        # Run inspect_forward on the complete sequence
        result = self.inspect_forward(curr_ids, tokenizer=tokenizer)
        result["prompt_len"] = prompt_len
        result["generated_tokens"] = generated_token_ids
        return result
