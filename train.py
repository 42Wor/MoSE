"""Local Training Script for MoSE (train.py).

Implements local training on NVIDIA Quadro T1000 (4 GB VRAM) featuring:
1. Zero-RAM Data Loading: Reads flat uint16 binary files via numpy.memmap
2. Automatic Mixed Precision: torch.amp.autocast('cuda', dtype=torch.float16)
3. Gradient Scaling: torch.amp.GradScaler('cuda') to prevent underflow
4. Cosine Learning Rate Schedule with Warmup
5. Gradient Accumulation & Parameter Checkpointing to mose_python_model.pt
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from config import Config
from model import MoSELanguageModel


class BinaryMemmapDataset:
    """Zero-copy memory-mapped dataset over flat 1D uint16 binary token arrays."""

    def __init__(self, bin_path: Path, seq_len: int):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Binary token file not found at '{self.bin_path}'. "
                "Please run data_pipeline/download_dataset.py -> preprocess_chunks.py -> tokenize_to_bin.py first."
            )

        # Open file as memory-mapped array in read-only mode
        self.data = np.memmap(str(self.bin_path), dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        assert self.total_tokens > self.seq_len + 1, (
            f"Dataset has {self.total_tokens} tokens, but requires at least {self.seq_len + 2}."
        )

    def get_batch(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples random sequences from the memory map without copying into system RAM."""
        # Pick random starting offsets
        high = self.total_tokens - self.seq_len - 1
        starts = np.random.randint(0, high, size=batch_size)

        x_list = []
        y_list = []
        for s in starts:
            # Slices read directly from memory-mapped page cache
            chunk = self.data[s : s + self.seq_len + 1].astype(np.int64)
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])

        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))

        if "cuda" in device:
            # Non-blocking async transfer to GPU memory
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        return x, y


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Computes cosine decay learning rate with linear warmup."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_val_loss(
    model: MoSELanguageModel,
    val_dataset: BinaryMemmapDataset,
    batch_size: int,
    eval_iters: int,
    device: str,
    use_fp16: bool,
) -> Tuple[float, float]:
    """Estimates validation task loss and auxiliary loss over eval_iters batches."""
    model.eval()
    task_losses = []
    aux_losses = []
    amp_device = "cuda" if "cuda" in device else "cpu"
    amp_enabled = use_fp16 and ("cuda" in device)

    for _ in range(eval_iters):
        x, y = val_dataset.get_batch(batch_size, device)
        with torch.amp.autocast(device_type=amp_device, dtype=torch.float16, enabled=amp_enabled):
            out = model(x, targets=y)
            task_losses.append(out["task_loss"].item())
            aux_losses.append(out["aux_loss"].item())

    model.train()
    return float(np.mean(task_losses)), float(np.mean(aux_losses))


def train_mose(
    config: Optional[Config] = None,
    checkpoint_out: Optional[str] = None,
    override_max_steps: Optional[int] = None,
    generate_dummy: bool = False,
):
    """Executes the full local training loop with mixed precision and memmap loading."""
    cfg = config or Config()
    device = cfg.device
    save_path = cfg.get_path(checkpoint_out or cfg.checkpoint_path)

    print("=" * 65)
    print("MoSE: Mixture of Smaller Experts Local Training")
    print(f"Target Device : {device.upper()}")
    if "cuda" in device:
        print(f"GPU Model     : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Precision     : FP16 AMP ({'Enabled' if cfg.use_fp16 else 'Disabled'})")
    print(f"Micro-Batch   : {cfg.batch_size} (Seq Len: {cfg.seq_len})")
    print(f"Grad Accum    : {cfg.gradient_accumulation_steps} steps (Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps})")
    print(f"Save Path     : {save_path}")
    print("=" * 65)

    train_bin = cfg.get_path(cfg.train_bin_path)
    val_bin = cfg.get_path(cfg.val_bin_path)

    # Optional dummy data generation for pipeline testing
    if generate_dummy or not train_bin.exists():
        if generate_dummy:
            print("\nGenerating synthetic dummy binary memmaps for verification...")
            train_bin.parent.mkdir(parents=True, exist_ok=True)
            dummy_tokens = np.random.randint(0, cfg.vocab_size, size=20000, dtype=np.uint16)
            dummy_tokens.tofile(str(train_bin))
            val_tokens = np.random.randint(0, cfg.vocab_size, size=5000, dtype=np.uint16)
            val_tokens.tofile(str(val_bin))
            print("Synthetic train.bin and val.bin created successfully.")

    # 1. Initialize Datasets via zero-copy memmaps
    
    print("\nInitializing memory-mapped data loaders...")
    train_dataset = BinaryMemmapDataset(train_bin, cfg.seq_len)
    val_dataset = BinaryMemmapDataset(val_bin, cfg.seq_len)
    print(f"Train Tokens  : {train_dataset.total_tokens:,}")
    print(f"Val Tokens    : {val_dataset.total_tokens:,}")

    # 2. Build MoSE Language Model
    print("\nInstantiating MoSE Language Model...")
    model = MoSELanguageModel(cfg)
    model.to(device)

    param_stats = model.count_parameters()
    print(f"Total Parameters     : {param_stats['total_parameters']:,}")
    print(f"Active per Token     : {param_stats['active_parameters_per_token']:,}")
    print(f"Expert Sparsity      : {param_stats['sparsity_ratio'] * 100:.1f}%")

    # 3. Optimizer & Scaler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        fused=True if "cuda" in device else False,
    )

    amp_enabled = cfg.use_fp16 and ("cuda" in device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    max_steps = override_max_steps or cfg.max_steps
    eval_interval = 200
    best_val_loss = float("inf")

    print("\nStarting training loop...")
    start_time = time.time()
    tokens_seen = 0
    step = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        t0 = time.time()
        # Learning rate schedule
        lr = get_lr(step, cfg.warmup_steps, max_steps, cfg.learning_rate, cfg.min_learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_task_loss = 0.0
        accum_aux_loss = 0.0

        # Gradient accumulation micro-steps
        for micro_step in range(cfg.gradient_accumulation_steps):
            x, y = train_dataset.get_batch(cfg.batch_size, device)

            with torch.amp.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(x, targets=y)
                loss = outputs["total_loss"] / cfg.gradient_accumulation_steps

            accum_task_loss += outputs["task_loss"].item() / cfg.gradient_accumulation_steps
            accum_aux_loss += outputs["aux_loss"].item() / cfg.gradient_accumulation_steps

            # Backward pass with scaler
            scaler.scale(loss).backward()
            tokens_seen += cfg.batch_size * cfg.seq_len

        # Gradient clipping & optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step_time = time.time() - t0
        tokens_per_sec = (cfg.batch_size * cfg.gradient_accumulation_steps * cfg.seq_len) / max(step_time, 1e-4)

        # Console logging
        if step % 20 == 0 or step == max_steps - 1:
            vram_info = ""
            if "cuda" in device:
                vram_used = torch.cuda.max_memory_allocated() / (1024**2)
                vram_info = f" | VRAM: {vram_used:4.0f} MB"
            print(
                f"Step {step:5d}/{max_steps} | "
                f"Task Loss: {accum_task_loss:.4f} | "
                f"Aux Loss: {accum_aux_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {tokens_per_sec:5.0f} tok/s"
                f"{vram_info}"
            )

        # Periodic Validation & Checkpointing
        if (step > 0 and step % eval_interval == 0) or (step == max_steps - 1):
            val_task_loss, val_aux_loss = estimate_val_loss(
                model, val_dataset, cfg.batch_size, eval_iters=25, device=device, use_fp16=cfg.use_fp16
            )
            print("-" * 65)
            print(
                f"[Validation @ Step {step}] Task Loss: {val_task_loss:.4f} | "
                f"Aux Loss: {val_aux_loss:.4f} | "
                f"Perplexity: {math.exp(min(val_task_loss, 20)):.2f}"
            )
            print("-" * 65)

            if val_task_loss < best_val_loss:
                best_val_loss = val_task_loss
                print(f"--> Lowest validation loss achieved ({best_val_loss:.4f}). Saving checkpoint to {save_path.name}...")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config_dict": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))},
                        "step": step,
                        "val_loss": val_task_loss,
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    str(save_path),
                )

        step += 1

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print("MoSE Training Completed Successfully!")
    print(f"Total Steps       : {step:,}")
    print(f"Total Tokens Seen : {tokens_seen:,}")
    print(f"Training Time     : {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Best Val Loss     : {best_val_loss:.4f}")
    print(f"Final Model Saved : {save_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local training script for MoSE on Quadro T1000.")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--checkpoint", type=str, default="mose_python_model.pt", help="Checkpoint filename")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cuda/cpu)")
    parser.add_argument("--dummy_data", action="store_true", help="Generate synthetic dummy memmap data for test run")

    args = parser.parse_args()
    cfg = Config()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.device:
        cfg.device = args.device

    train_mose(
        cfg,
        checkpoint_out=args.checkpoint,
        override_max_steps=args.max_steps,
        generate_dummy=args.dummy_data,
    )
