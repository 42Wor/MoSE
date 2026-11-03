"""Stage 1: Streaming Dataset Downloader (data_pipeline/download_dataset.py)

Streams raw Python code from Hugging Face Hub (bigcode/the-stack-smol or smollm-corpus)
and writes raw Parquet files to data/raw/ incrementally without accumulating records in RAM.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


def download_python_dataset(
    dataset_name: str = "bigcode/the-stack-smol",
    dataset_subset: str = "data/python",
    target_mb: float = 500.0,
    output_dir: str = "data/raw",
    batch_size: int = 2000,
) -> Path:
    """Streams Python source files from Hugging Face and saves directly to Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    cfg = Config()
    out_path = cfg.get_path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_bytes = int(target_mb * 1024 * 1024)
    print("=" * 65)
    print("Stage 1: Streaming Dataset Download to Raw Parquet")
    print(f"Target Size : {target_mb:.1f} MB ({target_bytes:,} bytes)")
    print(f"Source      : {dataset_name} [{dataset_subset}]")
    print(f"Output Dir  : {out_path}")
    print("=" * 65)

    print("\nConnecting to streaming dataset on Hugging Face...")
    ds = None
    first_item = None

    # Try primary dataset
    try:
        if dataset_subset:
            candidate_ds = load_dataset(dataset_name, data_dir=dataset_subset, split="train", streaming=True)
        else:
            candidate_ds = load_dataset(dataset_name, split="train", streaming=True)
        # Probe the stream to verify network & permissions
        ds_iter = iter(candidate_ds)
        first_item = next(ds_iter)
        ds = candidate_ds
        print(f"Connected successfully to {dataset_name} [{dataset_subset}]")
    except Exception as exc:
        print(f"Primary dataset {dataset_name} unavailable or gated: {exc}")
        print(f"Switching to open fallback dataset: {cfg.fallback_dataset} [{cfg.fallback_subset}]...")
        dataset_name = cfg.fallback_dataset
        dataset_subset = cfg.fallback_subset
        try:
            candidate_ds = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)
        except Exception:
            candidate_ds = load_dataset(dataset_name, split="train", streaming=True)
        ds_iter = iter(candidate_ds)
        first_item = next(ds_iter)
        ds = candidate_ds
        print(f"Connected successfully to {dataset_name} [{dataset_subset}]")

    # Define strict schema for fast binary parquet writing
    schema = pa.schema([
        ("content", pa.string()),
        ("byte_size", pa.int64()),
    ])

    shard_idx = 1
    parquet_file = out_path / f"python_raw_{shard_idx:03d}.parquet"
    writer = pq.ParquetWriter(str(parquet_file), schema, compression="SNAPPY")

    batch_contents = []
    batch_sizes = []
    total_bytes = 0
    total_files = 0
    start_time = time.time()

    try:
        for item in ds:
            # Extract code text regardless of column naming conventions
            code = item.get("content") or item.get("text") or item.get("code") or ""
            if not isinstance(code, str) or not code.strip():
                continue

            raw_bytes = len(code.encode("utf-8"))
            batch_contents.append(code)
            batch_sizes.append(raw_bytes)
            total_bytes += raw_bytes
            total_files += 1

            # Flush batch directly to disk
            if len(batch_contents) >= batch_size:
                table = pa.Table.from_arrays(
                    [pa.array(batch_contents, type=pa.string()), pa.array(batch_sizes, type=pa.int64())],
                    schema=schema,
                )
                writer.write_table(table)
                batch_contents.clear()
                batch_sizes.clear()

                elapsed = time.time() - start_time
                mb_done = total_bytes / (1024 * 1024)
                rate = mb_done / max(elapsed, 0.001)
                print(
                    f"\r[Streaming] Downloaded {mb_done:6.1f} / {target_mb:.1f} MB "
                    f"({total_files:,} files) | Speed: {rate:.2f} MB/s",
                    end="",
                    flush=True,
                )

            if total_bytes >= target_bytes:
                break

        # Flush residual rows
        if batch_contents:
            table = pa.Table.from_arrays(
                [pa.array(batch_contents, type=pa.string()), pa.array(batch_sizes, type=pa.int64())],
                schema=schema,
            )
            writer.write_table(table)
            batch_contents.clear()
            batch_sizes.clear()

    finally:
        writer.close()

    elapsed = time.time() - start_time
    file_disk_mb = parquet_file.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 65)
    print("Stage 1 Download Complete!")
    print(f"Total Text Ingested : {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Total Source Files  : {total_files:,}")
    print(f"Parquet Saved       : {parquet_file} ({file_disk_mb:.2f} MB)")
    print(f"Elapsed Time        : {elapsed:.1f}s")
    print("=" * 65 + "\n")
    return parquet_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Stream and save raw Python code into Parquet.")
    parser.add_argument("--target_mb", type=float, default=500.0, help="Target raw text size in MB (default: 500.0)")
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-smol", help="Hugging Face dataset identifier")
    parser.add_argument("--subset", type=str, default="data/python", help="Subset or data_dir in dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Target raw output directory")
    parser.add_argument("--batch_size", type=int, default=2000, help="Flush batch size for Parquet writing")

    args = parser.parse_args()
    download_python_dataset(
        dataset_name=args.dataset,
        dataset_subset=args.subset,
        target_mb=args.target_mb,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
