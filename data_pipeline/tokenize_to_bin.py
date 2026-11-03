"""Stage 3: Binary Tokenization & Memmap Preparation (data_pipeline/tokenize_to_bin.py)

Trains or loads a code-aware Byte-Pair Encoding (BPE) tokenizer (vocab = 10,000)
including FIM sentinel tokens (<PRE>, <SUF>, <MID>), tokenizes all JSONL chunks,
and flattens the sequences into continuous 1D uint16 binary files:
- data/binary/train.bin (90% split)
- data/binary/val.bin   (10% split)
- data/binary/meta.json (Vocab, split counts, token mapping)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def sample_text_iterator(chunks: List[Path], max_lines: int = 50000) -> Iterator[str]:
    """Iterates through JSONL chunks to supply training samples for BPE tokenizer."""
    count = 0
    for chunk_file in chunks:
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if text:
                        yield text
                        count += 1
                        if count >= max_lines:
                            return
                except json.JSONDecodeError:
                    continue


def get_or_train_tokenizer(
    chunks: List[Path],
    tokenizer_path: Path,
    vocab_size: int = 10000,
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """Trains a ByteLevel BPE tokenizer on sampled code or loads existing tokenizer."""
    if special_tokens is None:
        special_tokens = ["<PAD>", "<EOS>", "<PRE>", "<SUF>", "<MID>"]

    if tokenizer_path.exists():
        loaded = Tokenizer.from_file(str(tokenizer_path))
        if loaded.get_vocab_size() >= vocab_size:
            print(f"Loading existing tokenizer from {tokenizer_path} (vocab_size={loaded.get_vocab_size():,})...")
            return loaded
        else:
            print(f"Existing tokenizer at {tokenizer_path} has only {loaded.get_vocab_size():,} tokens (< target {vocab_size:,}). Retraining...")

    print(f"\nTraining new ByteLevel BPE tokenizer (vocab_size={vocab_size:,})...")
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    bpe = ByteLevelBPETokenizer()
    text_iter = sample_text_iterator(chunks, max_lines=40000)

    bpe.train_from_iterator(
        iterator=text_iter,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    # Save to file
    bpe.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path} with {bpe.get_vocab_size():,} tokens.\n")
    return bpe


def tokenize_chunks_to_binary(
    chunks_dir: str = "data/chunks",
    binary_dir: str = "data/binary",
    vocab_size: int = 10000,
    train_ratio: float = 0.9,
    buffer_tokens: int = 500000,
) -> Dict[str, any]:
    """Tokenizes all chunks and writes continuous 1D uint16 memory-mapped binary files."""
    cfg = Config()
    chunks_path = cfg.get_path(chunks_dir)
    binary_path = cfg.get_path(binary_dir)
    binary_path.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(list(chunks_path.glob("*.jsonl")))
    if not chunk_files:
        raise FileNotFoundError(f"No .jsonl chunks found in {chunks_path}. Run preprocess_chunks.py first.")

    tokenizer_path = binary_path / "tokenizer.json"
    tokenizer = get_or_train_tokenizer(
        chunk_files,
        tokenizer_path,
        vocab_size=vocab_size,
        special_tokens=cfg.special_tokens,
    )

    eos_token_id = tokenizer.token_to_id(cfg.eos_token)
    if eos_token_id is None:
        eos_token_id = 1

    train_bin_path = binary_path / "train.bin"
    val_bin_path = binary_path / "val.bin"

    print("=" * 65)
    print("Stage 3: Tokenization & Flat Binary Writing")
    print(f"Input Chunks  : {len(chunk_files)} files in {chunks_path}")
    print(f"Train File    : {train_bin_path} ({train_ratio * 100:.0f}%)")
    print(f"Val File      : {val_bin_path} ({(1 - train_ratio) * 100:.0f}%)")
    print(f"Format        : 1D uint16 array (memory-mapped ready)")
    print("=" * 65)

    train_f = open(train_bin_path, "wb")
    val_f = open(val_bin_path, "wb")

    train_tokens_count = 0
    val_tokens_count = 0

    train_buf = []
    val_buf = []

    # Deterministic 90/10 split per code document (10% validation, 90% training)
    n_chunks = len(chunk_files)
    val_interval = int(round(1.0 / (1.0 - train_ratio))) if train_ratio < 1.0 else 0
    doc_idx = 0

    try:
        for idx, chunk_path in enumerate(chunk_files):
            print(f"Tokenizing [{idx + 1}/{n_chunks}] {chunk_path.name}...")

            with open(chunk_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if not text:
                            continue

                        # Tokenize text
                        encoded = tokenizer.encode(text)
                        token_ids = encoded.ids
                        if not token_ids:
                            continue

                        # Append EOS token to delimit code documents
                        token_ids.append(eos_token_id)
                        doc_idx += 1

                        # Assign to train or val based on ratio
                        is_val = (val_interval > 0) and (doc_idx % val_interval == 0)
                        if is_val:
                            val_buf.extend(token_ids)
                            val_tokens_count += len(token_ids)
                        else:
                            train_buf.extend(token_ids)
                            train_tokens_count += len(token_ids)
                            train_buf.extend(token_ids)
                            train_tokens_count += len(token_ids)

                        # Write buffered binary data
                        if len(train_buf) >= buffer_tokens:
                            arr = np.array(train_buf, dtype=np.uint16)
                            train_f.write(arr.tobytes())
                            train_buf.clear()

                        if len(val_buf) >= buffer_tokens:
                            arr = np.array(val_buf, dtype=np.uint16)
                            val_f.write(arr.tobytes())
                            val_buf.clear()

                    except json.JSONDecodeError:
                        continue

        # Flush remaining buffers
        if train_buf:
            arr = np.array(train_buf, dtype=np.uint16)
            train_f.write(arr.tobytes())
            train_buf.clear()

        if val_buf:
            arr = np.array(val_buf, dtype=np.uint16)
            val_f.write(arr.tobytes())
            val_buf.clear()

    finally:
        train_f.close()
        val_f.close()

    total_tokens = train_tokens_count + val_tokens_count
    train_mb = (train_tokens_count * 2) / (1024 * 1024)
    val_mb = (val_tokens_count * 2) / (1024 * 1024)

    # Save metadata
    special_tokens_dict = {}
    for tok in cfg.special_tokens:
        tok_id = tokenizer.token_to_id(tok)
        if tok_id is not None:
            special_tokens_dict[tok] = tok_id

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": train_tokens_count,
        "val_tokens": val_tokens_count,
        "total_tokens": total_tokens,
        "dtype": "uint16",
        "special_tokens": special_tokens_dict,
        "train_bin_bytes": int(train_tokens_count * 2),
        "val_bin_bytes": int(val_tokens_count * 2),
    }

    meta_path = binary_path / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 65)
    print("Stage 3 Tokenization Complete!")
    print(f"Total Tokens Encoded: {total_tokens:,}")
    print(f"Train Tokens        : {train_tokens_count:,} ({train_mb:.2f} MB)")
    print(f"Val Tokens          : {val_tokens_count:,} ({val_mb:.2f} MB)")
    print(f"Metadata File       : {meta_path}")
    print("=" * 65 + "\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Tokenize chunks into fast binary memmaps.")
    parser.add_argument("--chunks_dir", type=str, default="data/chunks", help="Input directory of JSONL chunks")
    parser.add_argument("--binary_dir", type=str, default="data/binary", help="Output directory for binary files")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size (default: 10000)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio (default: 0.9)")

    args = parser.parse_args()
    tokenize_chunks_to_binary(
        chunks_dir=args.chunks_dir,
        binary_dir=args.binary_dir,
        vocab_size=args.vocab_size,
        train_ratio=args.train_ratio,
    )

