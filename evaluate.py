"""Evaluation & Expert Analysis Suite for MoSE (evaluate.py).

Performs two core benchmark evaluations:
1. Code Infilling (FIM): Autoregressive Fill-in-the-Middle code completion
   given '<PRE>def add(a, b): <SUF> return result <MID>'.
2. Expert Utilization Heatmap: Passes Python keywords (def, import, for, class, return)
   through MoSE layers and renders an ASCII matrix showing which micro-experts specialize
   for distinct semantic syntactic tokens.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer

# Ensure clean UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import Config
from model import MoSELanguageModel


def load_model_and_tokenizer(
    checkpoint_path: str = "mose_python_model.pt",
    tokenizer_path: str = "data/binary/tokenizer.json",
    device: Optional[str] = None,
) -> Tuple[MoSELanguageModel, Tokenizer, Config]:
    """Loads trained checkpoint or initializes model for evaluation."""
    cfg = Config()
    if device is not None:
        cfg.device = device

    chk_file = cfg.get_path(checkpoint_path)
    tok_file = cfg.get_path(tokenizer_path)

    # 1. Load Tokenizer
    if not tok_file.exists():
        print(f"Warning: Tokenizer not found at {tok_file}. Initializing fallback ByteLevel tokenizer...")
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(["def add(a, b):\n    result = a + b\n    return result\nimport os\nfor i in range(10):\n    pass\nclass Test:\n    pass\n"], vocab_size=cfg.vocab_size, special_tokens=cfg.special_tokens)
        tok_file.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(tok_file))
    else:
        tok = Tokenizer.from_file(str(tok_file))

    # 2. Load Model
    model = MoSELanguageModel(cfg)
    if chk_file.exists():
        print(f"Loading checkpoint weights from {chk_file}...")
        try:
            checkpoint = torch.load(str(chk_file), map_location=cfg.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(chk_file), map_location=cfg.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from Step {checkpoint.get('step', 0)} (Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Notice: No checkpoint found at {chk_file}. Running evaluation with freshly initialized MoSE weights.")

    model.to(cfg.device)
    model.eval()
    return model, tok, cfg


# ==============================================================================
# Test 1: Fill-in-the-Middle (FIM) Code Infilling
# ==============================================================================

def test_code_infilling(
    model: MoSELanguageModel,
    tokenizer: Tokenizer,
    config: Config,
    prompt: Optional[str] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
):
    """Executes Fill-in-the-Middle (FIM) code completion."""
    if prompt is None:
        prompt = "<PRE>def add(a, b):\n    <SUF>\n    return result\n<MID>"

    print("\n" + "=" * 65)
    print("TEST 1: Fill-in-the-Middle (FIM) Code Infilling")
    print("=" * 65)
    print(f"Prompt:\n{prompt}\n")

    encoded = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([encoded.ids], dtype=torch.long, device=config.device)

    eos_id = tokenizer.token_to_id(config.eos_token)
    suf_id = tokenizer.token_to_id(config.fim_suf_token)

    with torch.no_grad():
        generated_ids = model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            eos_id=eos_id,
        )

    full_output = tokenizer.decode(generated_ids[0].tolist())
    new_tokens = generated_ids[0][len(encoded.ids):].tolist()
    generated_middle = tokenizer.decode(new_tokens)

    print("-" * 65)
    print(f"Generated Infilled Middle:\n{generated_middle.strip()}")
    print("-" * 65)
    print(f"Full Model Sequence:\n{full_output}")
    print("=" * 65 + "\n")


# ==============================================================================
# Test 2: Expert Utilization Heatmap Across Python Keywords
# ==============================================================================

def test_expert_utilization(
    model: MoSELanguageModel,
    tokenizer: Tokenizer,
    config: Config,
    keywords: Optional[List[str]] = None,
):
    """Measures micro-expert activation across syntactic Python keywords."""
    if keywords is None:
        keywords = ["def", "import", "for", "class", "return", "try", "lambda", "assert"]

    print("=" * 65)
    print("TEST 2: Micro-Expert Routing Heatmap Across Python Keywords")
    print(f"Layers: {config.n_layers} | Experts per Layer: {config.num_experts} (Top-{config.top_k} routing)")
    print("=" * 65)

    num_experts = config.num_experts
    # Matrix: [len(keywords), num_experts]
    activation_counts = np.zeros((len(keywords), num_experts), dtype=np.int32)

    with torch.no_grad():
        for k_idx, kw in enumerate(keywords):
            # Tokenize keyword (with a leading space to match code tokens)
            encoded = tokenizer.encode(f" {kw}")
            if not encoded.ids:
                encoded = tokenizer.encode(kw)

            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=config.device)
            outputs = model(input_ids)
            routing_layers = outputs["routing_indices"]  # List of [N_tokens, top_k]

            # Aggregate expert selections across all layers for these keyword tokens
            for layer_routing in routing_layers:
                # layer_routing shape: [N_tokens, top_k]
                indices = layer_routing.cpu().numpy().flatten()
                for exp_id in indices:
                    activation_counts[k_idx, exp_id] += 1

    # Print ASCII Heatmap
    header = f"{'Keyword':<10} | " + " ".join([f"E{e:<2}" for e in range(num_experts)])
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    max_val = np.max(activation_counts) if np.max(activation_counts) > 0 else 1

    # Visual intensity glyphs (ASCII-safe for Windows command prompts)
    glyphs = [" . ", " * ", " # ", "###"]

    for k_idx, kw in enumerate(keywords):
        row_str = f"{kw:<10} | "
        for exp_id in range(num_experts):
            val = activation_counts[k_idx, exp_id]
            norm_val = val / max_val
            glyph_idx = min(int(norm_val * len(glyphs)), len(glyphs) - 1)
            row_str += glyphs[glyph_idx]
        print(row_str)

    print(sep)
    print("Glyph Legend: ' . ' = Inactive  |  ' * ' = Low  |  ' # ' = Medium  |  '###' = Dominant Specialization")

    # Routing Entropy Analysis
    row_sums = activation_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = activation_counts / row_sums
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    max_entropy = np.log(num_experts)

    print("\nExpert Specialization Diversity (Normalized Entropy):")
    for k_idx, kw in enumerate(keywords):
        norm_entropy = (entropy[k_idx] / max_entropy) * 100
        specialized = "High Specialization" if norm_entropy < 70 else "Distributed Load"
        print(f"  {kw:<10} : {norm_entropy:5.1f}% entropy ({specialized})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MoSE model on FIM and expert routing.")
    parser.add_argument("--checkpoint", type=str, default="mose_python_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/binary/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on")
    parser.add_argument("--prompt", type=str, default=None, help="Custom FIM prompt")

    args = parser.parse_args()
    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    test_code_infilling(model, tokenizer, config, prompt=args.prompt)
    test_expert_utilization(model, tokenizer, config)
