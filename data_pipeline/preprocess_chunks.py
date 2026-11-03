"""Stage 2: Code Preprocessing & FIM Sharding (data_pipeline/preprocess_chunks.py)

Reads raw Parquet files from data/raw/, filters out malformed or overly large files,
applies Fill-in-the-Middle (FIM) transformations to 50% of files, and writes
clean text shards into data/chunks/chunk_XXX.jsonl (~50 MB per shard).
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def apply_fim_transformation(
    code: str,
    pre_token: str = "<PRE>",
    suf_token: str = "<SUF>",
    mid_token: str = "<MID>",
) -> str:
    """Applies Fill-in-the-Middle (FIM) transformation in PSM format.
    
    Transforms: prefix + middle + suffix -> <PRE>prefix<SUF>suffix<MID>middle
    Split points are chosen uniformly at random across character positions.
    """
    n = len(code)
    if n < 30:
        return code

    # Sample two distinct split points
    cut1, cut2 = sorted(random.sample(range(1, n), 2))
    prefix = code[:cut1]
    middle = code[cut1:cut2]
    suffix = code[cut2:]

    return f"{pre_token}{prefix}{suf_token}{suffix}{mid_token}{middle}"


def is_valid_python_file(code: str, max_file_kb: float = 50.0, max_line_len: int = 1000) -> bool:
    """Filters code files based on length, runaway lines, and basic validity."""
    if not isinstance(code, str) or not code.strip():
        return False

    # Check maximum file size
    if len(code.encode("utf-8")) > max_file_kb * 1024:
        return False

    # Check for runaway lines (minified code, base64 strings, or data tables)
    lines = code.split("\n")
    for line in lines:
        if len(line) > max_line_len:
            return False

    return True


def preprocess_and_shard(
    raw_dir: str = "data/raw",
    chunks_dir: str = "data/chunks",
    chunk_size_mb: float = 50.0,
    fim_rate: float = 0.5,
    seed: int = 42,
) -> int:
    """Iterates through Parquet files in raw_dir, processes code, and writes JSONL shards."""
    import pyarrow.parquet as pq

    random.seed(seed)
    cfg = Config()
    raw_path = cfg.get_path(raw_dir)
    chunks_path = cfg.get_path(chunks_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(list(raw_path.glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {raw_path}. Run download_dataset.py first.")

    print("=" * 65)
    print("Stage 2: Code Preprocessing & FIM Sharding")
    print(f"Input Parquet files : {len(parquet_files)}")
    print(f"Target Chunk Size   : {chunk_size_mb} MB per JSONL shard")
    print(f"FIM Transformation  : {fim_rate * 100:.0f}% of files")
    print(f"Output Chunks Dir   : {chunks_path}")
    print("=" * 65)

    max_chunk_bytes = int(chunk_size_mb * 1024 * 1024)
    chunk_idx = 0
    current_chunk_path = chunks_path / f"chunk_{chunk_idx:03d}.jsonl"
    current_writer = open(current_chunk_path, "w", encoding="utf-8")
    current_chunk_bytes = 0

    total_inspected = 0
    total_valid = 0
    total_dropped = 0
    total_fim_applied = 0

    try:
        for p_file in parquet_files:
            print(f"\nProcessing Parquet file: {p_file.name}...")
            pq_file = pq.ParquetFile(str(p_file))

            # Iterate by row group to stream without memory spikes
            for i in range(pq_file.num_row_groups):
                table = pq_file.read_row_group(i, columns=["content"])
                contents = table.column("content").to_pylist()

                for raw_code in contents:
                    total_inspected += 1

                    if not is_valid_python_file(raw_code, cfg.max_file_size_kb, cfg.max_line_len):
                        total_dropped += 1
                        continue

                    total_valid += 1

                    # Apply FIM to designated fraction
                    if random.random() < fim_rate:
                        processed_text = apply_fim_transformation(
                            raw_code,
                            pre_token=cfg.fim_pre_token,
                            suf_token=cfg.fim_suf_token,
                            mid_token=cfg.fim_mid_token,
                        )
                        total_fim_applied += 1
                    else:
                        processed_text = raw_code

                    # Serialize to JSONL
                    line = json.dumps({"text": processed_text}, ensure_ascii=False) + "\n"
                    line_bytes = len(line.encode("utf-8"))

                    current_writer.write(line)
                    current_chunk_bytes += line_bytes

                    # Rotate chunk if size threshold exceeded
                    if current_chunk_bytes >= max_chunk_bytes:
                        current_writer.close()
                        print(f"  Saved {current_chunk_path.name} ({current_chunk_bytes / (1024 * 1024):.2f} MB)")
                        chunk_idx += 1
                        current_chunk_path = chunks_path / f"chunk_{chunk_idx:03d}.jsonl"
                        current_writer = open(current_chunk_path, "w", encoding="utf-8")
                        current_chunk_bytes = 0

    finally:
        current_writer.close()
        # Clean up empty chunk file if trailing rotation didn't write anything
        if current_chunk_path.exists() and current_chunk_path.stat().st_size == 0:
            current_chunk_path.unlink()
        else:
            print(f"  Saved {current_chunk_path.name} ({current_chunk_bytes / (1024 * 1024):.2f} MB)")
            chunk_idx += 1

    print("\n" + "=" * 65)
    print("Stage 2 Preprocessing Complete!")
    print(f"Files Inspected   : {total_inspected:,}")
    print(f"Files Accepted    : {total_valid:,}")
    print(f"Files Dropped     : {total_dropped:,} ({(total_dropped / max(total_inspected, 1)) * 100:.1f}%)")
    print(f"FIM Transformed   : {total_fim_applied:,} ({(total_fim_applied / max(total_valid, 1)) * 100:.1f}%)")
    print(f"Total Shards Made : {chunk_idx}")
    print("=" * 65 + "\n")
    return chunk_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Filter raw code and shard into JSONL with FIM.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing raw Parquet files")
    parser.add_argument("--chunks_dir", type=str, default="data/chunks", help="Output directory for JSONL shards")
    parser.add_argument("--chunk_size_mb", type=float, default=50.0, help="Shard size in MB")
    parser.add_argument("--fim_rate", type=float, default=0.5, help="Proportion of files to FIM transform")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    preprocess_and_shard(
        raw_dir=args.raw_dir,
        chunks_dir=args.chunks_dir,
        chunk_size_mb=args.chunk_size_mb,
        fim_rate=args.fim_rate,
        seed=args.seed,
    )

