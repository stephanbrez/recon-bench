"""
``rb eval-pcd`` — evaluate point clouds against a reference.

Compares predicted point clouds (e.g. NeRF, 3DGS exports) against a
dense reference cloud (e.g. COLMAP fused.ply). Crops predictions to the
reference AABB, downsamples to common voxel sizes, and reports Chamfer,
Hausdorff, and F-score metrics.

Assumes all clouds are already in the same coordinate frame.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

from typing import Iterable


# ===== Data Structures =====

_CLI_TO_REGISTRY = {
    "chamfer": "chamfer_distance",
    "hausdorff": "hausdorff_distance",
    "fscore": "fscore",
}

_AVAILABLE_METRICS = tuple(_CLI_TO_REGISTRY.keys())


@dataclasses.dataclass(slots=True)
class EvalRow:
    """Single evaluation result row for one method/voxel/threshold combo."""

    method: str
    voxel_size: float
    threshold: float
    ref_points: int
    pred_points: int
    chamfer: float | None = None
    hausdorff: float | None = None
    fscore: float | None = None


# ===== Helpers =====

def _parse_pred_specs(pred_specs: Iterable[str]) -> dict[str, str]:
    """
    Parse ``name=path`` prediction specs from the CLI.

    Parameters
    ----------
    pred_specs : Iterable[str]
        Each element has the form ``name=path``.

    Returns
    -------
    dict[str, str]
        Mapping of method name to file path.

    Raises
    ------
    ValueError
        If a spec is malformed.
    """
    out: dict[str, str] = {}
    for spec in pred_specs:
        if "=" not in spec:
            raise ValueError(
                f"Bad --pred spec '{spec}'. Expected name=path"
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(
                f"Bad --pred spec '{spec}'. Expected name=path"
            )
        out[name] = path
    return out


def _print_summary(rows: list[EvalRow]) -> None:
    """Print a grouped summary table to stdout."""
    grouped: dict[str, dict[float, list[EvalRow]]] = {}
    for r in rows:
        grouped.setdefault(r.method, {}).setdefault(
            r.voxel_size, [],
        ).append(r)

    for method in sorted(grouped):
        print(f"\n=== {method} ===")
        for voxel in sorted(grouped[method]):
            group = sorted(
                grouped[method][voxel], key=lambda x: x.threshold,
            )
            first = group[0]

            parts = [
                f"voxel={voxel:.6g}",
                f"ref_points={first.ref_points}",
                f"pred_points={first.pred_points}",
            ]
            if first.chamfer is not None:
                parts.append(f"chamfer={first.chamfer:.6g}")
            if first.hausdorff is not None:
                parts.append(f"hausdorff={first.hausdorff:.6g}")
            print(" | ".join(parts))

            if first.fscore is not None:
                for r in group:
                    print(
                        f"  F-score @ {r.threshold:.6g}: "
                        f"{r.fscore:.4f}"
                    )


# ===== CLI Registration =====

def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``eval-pcd`` subcommand."""
    parser = subparsers.add_parser(
        "eval-pcd",
        help="Evaluate point clouds against a reference.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r", "--reference", type=pathlib.Path, required=True,
        help="Reference cloud, e.g. COLMAP fused.ply.",
    )
    parser.add_argument(
        "-m", "--metrics", nargs="*", default=None,
        help="Metrics to compute (default: all). "
             "Options: chamfer, hausdorff, fscore.",
    )
    parser.add_argument(
        "-p", "--pred", nargs="+", required=True,
        help=(
            "Prediction clouds as name=path, "
            "e.g. nerf=nerf.ply 3dgs=gs.ply."
        ),
    )
    parser.add_argument(
        "-v", "--voxel-sizes", nargs="+", type=float,
        default=[0.01, 0.02, 0.05],
        help="Voxel sizes for evaluation (default: 0.01 0.02 0.05).",
    )
    parser.add_argument(
        "-f", "--fscore-mults", nargs="+", type=float,
        default=[1.0, 2.0, 4.0],
        help="F-score radii as multiples of voxel size "
             "(default: 1.0 2.0 4.0).",
    )
    parser.add_argument(
        "--crop-pad", type=float, default=0.0,
        help="Pad added to reference AABB before cropping.",
    )
    parser.add_argument(
        "--remove-outliers", action="store_true",
        help="Apply statistical outlier removal before downsampling.",
    )
    parser.add_argument(
        "--nb-neighbors", type=int, default=20,
        help="Neighbors for statistical outlier removal (default: 20).",
    )
    parser.add_argument(
        "--std-ratio", type=float, default=2.0,
        help="Std ratio for statistical outlier removal (default: 2.0).",
    )
    parser.add_argument(
        "--write-csv", type=pathlib.Path, default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--write-json", type=pathlib.Path, default=None,
        help="Optional JSON output path.",
    )
    parser.set_defaults(func=run)


# ===== Subcommand Entry Point =====

def run(args: argparse.Namespace) -> None:
    """Execute the ``eval-pcd`` subcommand."""
    from .. import _types
    from ..io import export
    from ..io import geometry
    from ..metrics import geometry as metrics_geometry
    from ..utils import pointcloud

    # ─── Resolve requested metrics ───
    requested_cli = (
        list(_AVAILABLE_METRICS) if args.metrics is None else args.metrics
    )
    unknown = [m for m in requested_cli if m not in _CLI_TO_REGISTRY]
    if unknown:
        print(
            f"Error: unknown metric(s): {unknown}. "
            f"Available: {list(_AVAILABLE_METRICS)}"
        )
        sys.exit(1)

    registry_names = [_CLI_TO_REGISTRY[m] for m in requested_cli]
    use_fscore = "fscore" in requested_cli

    pred_map = _parse_pred_specs(args.pred)

    # ─── Load and preprocess reference ───
    ref_path: pathlib.Path = args.reference
    if not ref_path.exists():
        print(f"Error: reference file not found: {ref_path}")
        sys.exit(1)

    print(f"Loading reference: {ref_path}")
    ref = geometry.load_point_cloud(ref_path)
    ref_bbox = pointcloud.compute_bounding_box(ref, pad=args.crop_pad)
    diag = pointcloud.bounding_box_diagonal(ref_bbox)
    print(f"Reference points: {pointcloud.num_points(ref)}")
    print(f"Reference AABB diagonal: {diag:.6g}")

    print("Cropping reference to its own AABB...")
    ref = pointcloud.crop_to_bounding_box(ref, ref_bbox)

    if args.remove_outliers:
        print("Removing outliers from reference...")
        ref = pointcloud.remove_outliers(
            ref, args.nb_neighbors, args.std_ratio,
        )

    # ─── Load and preprocess predictions ───
    pred_clouds: dict[str, object] = {}
    for name, path_str in pred_map.items():
        pred_path = pathlib.Path(path_str)
        if not pred_path.exists():
            print(f"Error: prediction file not found: {pred_path}")
            sys.exit(1)

        print(f"Loading prediction: {name} -> {pred_path}")
        pcd = geometry.load_point_cloud(pred_path)
        pcd = pointcloud.crop_to_bounding_box(pcd, ref_bbox)
        if args.remove_outliers:
            pcd = pointcloud.remove_outliers(
                pcd, args.nb_neighbors, args.std_ratio,
            )
        pred_clouds[name] = pcd

    # ─── Evaluate at each voxel size ───
    rows: list[EvalRow] = []

    for voxel_size in args.voxel_sizes:
        thresholds = [voxel_size * m for m in args.fscore_mults]

        print(
            f"\nEvaluating voxel_size={voxel_size:.6g}, "
            f"thresholds={thresholds}"
        )
        ref_ds = pointcloud.voxel_downsample(ref, voxel_size)
        ref_n = pointcloud.num_points(ref_ds)

        for name, pred in pred_clouds.items():
            pred_ds = pointcloud.voxel_downsample(pred, voxel_size)
            pred_n = pointcloud.num_points(pred_ds)

            scores = metrics_geometry.compute_geometry_metrics(
                ref_ds,
                pred_ds,
                metrics=registry_names,
                mode=_types.GeometryType.POINTCLOUD,
                thresholds=thresholds,
            )

            chamfer_val = (
                float(scores["chamfer_distance"].item())
                if "chamfer_distance" in scores else None
            )
            hausdorff_val = (
                float(scores["hausdorff_distance"].item())
                if "hausdorff_distance" in scores else None
            )
            fscores = (
                scores["fscore"].tolist()
                if "fscore" in scores else []
            )

            # One row per threshold (fscore varies); chamfer/hausdorff
            # are the same across thresholds so they repeat.
            row_thresholds = thresholds if fscores else [thresholds[0]]
            for i, thr in enumerate(row_thresholds):
                rows.append(
                    EvalRow(
                        method=name,
                        voxel_size=voxel_size,
                        threshold=thr,
                        ref_points=ref_n,
                        pred_points=pred_n,
                        chamfer=chamfer_val,
                        hausdorff=hausdorff_val,
                        fscore=fscores[i] if fscores else None,
                    ),
                )

    # ─── Output ───
    _print_summary(rows)

    data = [dataclasses.asdict(r) for r in rows]

    if args.write_csv:
        export.write_to_csv(args.write_csv, data)
        print(f"\nWrote CSV: {args.write_csv}")

    if args.write_json:
        export.write_to_json(args.write_json, data)
        print(f"Wrote JSON: {args.write_json}")
