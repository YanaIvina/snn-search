#!/usr/bin/env python3
"""
Build and save a local cached Google Speech Commands dataset inside the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from speech_dataset import (
    create_dataset,
    default_cache_path,
    resolve_raw_data_path,
    save_dataset_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute mel-spectrogram features for Google Speech Commands and save them as a local .npz cache."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to an existing raw Google Speech Commands dataset directory. If omitted, kagglehub is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the cached dataset. Defaults to `data/google_speech_commands_mels.npz` in the repo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def main() -> int:
    args = parse_args()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_cache_path().resolve()
    )
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Cached dataset already exists: {output_path}. Use --force to overwrite it."
        )

    raw_data_path = resolve_raw_data_path(args.data_path)
    print(f"Raw dataset path: {raw_data_path}")

    X, y, label_map = create_dataset(raw_data_path)
    print(f"Dataset built: X={X.shape}, y={y.shape}, classes={len(label_map)}")

    saved_path = save_dataset_cache(
        output_path,
        X,
        y,
        label_map,
        source_path=raw_data_path,
    )
    print(f"Cached dataset saved to: {saved_path}")
    print(f"Cache size: {format_size(saved_path.stat().st_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
