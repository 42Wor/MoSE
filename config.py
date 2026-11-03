"""Global configuration for Mixture of Smaller Experts (MoSE).

Defines architectural dimensions, training hyperparameters, and filesystem paths
optimized for training within 2.5 GB peak VRAM on an NVIDIA Quadro T1000 (4 GB).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # --------------------------------------------------------------------------
    # Architecture Dimensions
    # --------------------------------------------------------------------------
    vocab_size: int = 10000                 # Lightweight code-specialized BPE vocab
    d_model: int = 192                      # Model embedding dimension
    n_heads: int = 6                        # Attention heads (head_dim = 192 // 6 = 32)
    n_layers: int = 4                       # Transformer depth
    seq_len: int = 256                      # Context window length

    # --------------------------------------------------------------------------
    # Mixture of Smaller Experts (MoSE) Settings
    # --------------------------------------------------------------------------
    num_experts: int = 16                   # 16 fine-grained micro-experts per layer
    expert_dim: int = 128                   # Intermediate FFN hidden dimension
    top_k: int = 2                          # Dynamic routing activates top-2 experts
    has_shared_expert: bool = True          # 1 persistent shared base expert
    shared_expert_dim: int = 128            # Shared expert intermediate dimension
    aux_loss_weight: float = 0.01           # Switch Transformer load balancing coefficient

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    batch_size: int = 16                    # Micro-batch size (16 * 256 = 4,096 tokens)
    gradient_accumulation_steps: int = 2    # Effective batch = 32 (8,192 tokens/step)
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5         # Cosine schedule floor
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 200
    epochs: int = 3
    max_steps: int = 15000                  # Max training steps safety ceiling

    # Hardware & Precision
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True                   # torch.amp.autocast with FP16

    # --------------------------------------------------------------------------
    # Dataset & Tokenizer Paths
    # --------------------------------------------------------------------------
    raw_data_dir: str = "data/raw"
    chunks_dir: str = "data/chunks"
    binary_dir: str = "data/binary"
    train_bin_path: str = "data/binary/train.bin"
    val_bin_path: str = "data/binary/val.bin"
    meta_path: str = "data/binary/meta.json"
    tokenizer_path: str = "data/binary/tokenizer.json"
    checkpoint_path: str = "mose_python_model.pt"

    # Dataset Source
    dataset_name: str = "bigcode/the-stack-smol"
    dataset_subset: str = "data/python"
    fallback_dataset: str = "HuggingFaceTB/smollm-corpus"
    fallback_subset: str = "python-edu"

    # Ingestion & Chunking thresholds
    target_download_mb: float = 500.0       # Target download volume (500 MB - 1 GB)
    chunk_size_mb: float = 50.0             # Size per JSONL chunk
    max_file_size_kb: float = 50.0          # Drop files > 50 KB
    max_line_len: int = 1000                # Drop files with runaway lines > 1000 chars

    # --------------------------------------------------------------------------
    # Fill-in-the-Middle (FIM) & Special Tokens
    # --------------------------------------------------------------------------
    fim_rate: float = 0.5                   # 50% of files receive FIM transformation
    fim_pre_token: str = "<PRE>"
    fim_suf_token: str = "<SUF>"
    fim_mid_token: str = "<MID>"
    pad_token: str = "<PAD>"
    eos_token: str = "<EOS>"

    @property
    def special_tokens(self) -> List[str]:
        return [
            self.pad_token,
            self.eos_token,
            self.fim_pre_token,
            self.fim_suf_token,
            self.fim_mid_token,
        ]

    def get_path(self, rel_path: str) -> Path:
        return BASE_DIR / rel_path
