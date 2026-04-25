"""
``rb eval-images`` — batch image-vs-image evaluation.

Compares images in a target directory against images in a prediction
directory, paired by sorted index, then prints mean and per-image
metric tables.

By default, images are sorted by name and paired positionally. Use
``--match-names`` to pair by filename stem instead (ignoring extension),
skipping any unmatched files.
Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .tif, .webp.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
}


def _collect_images(directory: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted image paths from *directory*."""
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda p: p.name)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``eval-images`` subcommand."""
    parser = subparsers.add_parser(
        "eval-images",
        help="Batch image-vs-image evaluation.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-t", "--target", type=pathlib.Path, required=True,
        help="Directory of ground-truth images.",
    )
    parser.add_argument(
        "-p", "--prediction", type=pathlib.Path, required=True,
        help="Directory of predicted / reconstructed images.",
    )
    parser.add_argument(
        "-m", "--metrics", nargs="*", default=None,
        help="Metrics to compute (default: all). "
             "Options: psnr, ssim, ssim_windowed, lpips.",
    )
    parser.add_argument(
        "-P", "--profile", action="store_true",
        help="Enable profiling (timing + GPU memory).",
    )
    parser.add_argument(
        "-s", "--summary-only", action="store_true",
        help="Show only mean metrics, suppress per-item detail.",
    )
    parser.add_argument(
        "-M", "--match-names", action="store_true",
        help="Match target/prediction pairs by filename stem instead "
             "of sorted position. Unmatched files are skipped.",
    )
    parser.add_argument(
        "-S", "--shard-size", type=int, default=10, metavar="N",
        help="Max images per shard for metric computation (default: 10). "
             "Reduce if running out of GPU memory.",
    )
    parser.add_argument(
        "-r", "--resize", type=int, default=None, metavar="N",
        help="Downscale images so the longest edge is at most N pixels before "
             "computing metrics. Images smaller than N are unchanged. "
             "Useful for reducing VRAM usage with high-resolution inputs.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``eval-images`` subcommand."""
    import recon_bench

    # ─── Discover images ───
    target_dir: pathlib.Path = args.target
    pred_dir: pathlib.Path = args.prediction

    if not target_dir.is_dir():
        print(f"Error: target directory not found: {target_dir}")
        sys.exit(1)
    if not pred_dir.is_dir():
        print(f"Error: prediction directory not found: {pred_dir}")
        sys.exit(1)

    target_images = _collect_images(target_dir)
    if not target_images:
        print(f"Error: no images found in {target_dir}")
        sys.exit(1)

    pred_images = _collect_images(pred_dir)
    if not pred_images:
        print(f"Error: no images found in {pred_dir}")
        sys.exit(1)

    # ─── Pair images ───
    if args.match_names:
        pred_by_stem = {p.stem: p for p in pred_images}
        targets: list[pathlib.Path] = []
        predictions: list[pathlib.Path] = []
        skipped: list[str] = []
        for tgt in target_images:
            pred = pred_by_stem.get(tgt.stem)
            if pred is None:
                skipped.append(tgt.stem)
                continue
            targets.append(tgt)
            predictions.append(pred)
        if skipped:
            print(
                f"Warning: {len(skipped)} target image(s) had no "
                f"matching prediction: {', '.join(skipped[:5])}"
                + (f" ... and {len(skipped) - 5} more"
                   if len(skipped) > 5 else "")
            )
    else:
        n_pairs = min(len(target_images), len(pred_images))
        if len(target_images) != len(pred_images):
            print(
                f"Warning: image count mismatch — "
                f"{len(target_images)} target images, "
                f"{len(pred_images)} prediction images. "
                f"Evaluating first {n_pairs} pairs."
            )
        targets = target_images[:n_pairs]
        predictions = pred_images[:n_pairs]

    if not targets:
        print("Error: no image pairs to evaluate")
        sys.exit(1)

    print(f"Evaluating {len(targets)} image pairs ...\n")

    # ─── Run evaluation ───
    result = recon_bench.evaluate(
        targets,
        predictions,
        image_metrics=args.metrics,
        profile=args.profile,
        shard_size=args.shard_size,
        max_size=args.resize,
    )

    # ─── Display results ───
    print(result.summary())
    if not args.summary_only:
        detail = result.detail()
        if detail:
            print()
            print(detail)
