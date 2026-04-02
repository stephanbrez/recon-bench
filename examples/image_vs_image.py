"""
Example: batch image-vs-image evaluation using the library directly.

For production use, prefer the CLI:

    rb eval-images --target data/gt/ --prediction data/pred/

This script demonstrates how to call the library API and format
results programmatically.
"""
from __future__ import annotations

import pathlib

import recon_bench

_IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
}

# ─── Configure paths ───
TARGET_DIR = pathlib.Path("data/gt")
PREDICTION_DIR = pathlib.Path("data/pred")


def _collect_images(directory: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted image paths from *directory*."""
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda p: p.name)


def main() -> None:
    targets = _collect_images(TARGET_DIR)
    predictions = _collect_images(PREDICTION_DIR)

    n_pairs = min(len(targets), len(predictions))
    targets = targets[:n_pairs]
    predictions = predictions[:n_pairs]
    filenames = [p.stem for p in targets]

    result = recon_bench.evaluate(targets, predictions)
    print(result.summary(filenames=filenames))


if __name__ == "__main__":
    main()
