"""
recon-bench CLI entry point.

Provides the ``rb`` command with subcommands for evaluation tasks.

Usage
-----
    rb eval-images --target data/gt/ --prediction data/pred/
    rb --help
"""
from __future__ import annotations

import argparse

from . import eval_images


def main() -> None:
    """Top-level ``rb`` CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="rb",
        description="recon-bench: 3D reconstruction benchmarking toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ─── Register subcommands ───
    eval_images.register(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)
