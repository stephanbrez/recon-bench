from __future__ import annotations
import torch

from . import core
from .. import _types

from typing import Callable

# Registry mapping metric names to their core functions.
# Each function must accept (target, data, **kwargs) and return float | list[float].
_METRIC_REGISTRY: dict[str, Callable] = {
    "chamfer_distance": core.chamfer_distance,
    "hausdorff_distance": core.hausdorff_distance,
    "fscore": core.fscore,
}

AVAILABLE_METRICS = tuple(_METRIC_REGISTRY.keys())


def compute_geometry_metrics(
    target: _types.MeshInput | list[_types.MeshInput],
    data: _types.MeshInput | list[_types.MeshInput],
    metrics: list[str] | None = None,
    mode: _types.GeometryType = _types.GeometryType.MESH,
    num_points: int = 10000,
    thresholds: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute geometry metrics between target and predicted meshes or point clouds.

    Accepts single items or lists for batched evaluation. Returns per-item
    scores as tensors.

    Parameters
    ----------
    target : MeshInput or list[MeshInput]
        Ground truth geometry/geometries. Each item is a pathlib.Path or
        GeometryArrays dict.
    data : MeshInput or list[MeshInput]
        Predicted geometry/geometries. Must match the batch size of target.
    metrics : list[str] or None
        Names of metrics to compute. None computes all available metrics:
        "chamfer_distance", "hausdorff_distance", "fscore".
    mode : GeometryType
        Whether inputs are meshes or point clouds.
    num_points : int
        Number of surface sample points for distance computation.
    thresholds : list[float] or None
        F-score radii. Only used when "fscore" is in the requested metrics.
        Defaults to [0.01] if not provided.

    Returns
    -------
    dict[str, torch.Tensor]
        Metric name → per-item scores, shape (N,). Call ``.mean()`` to
        reduce to a scalar.

    Raises
    ------
    ValueError
        If an unknown metric name is requested, or if batch sizes differ.

    Examples
    --------
    >>> scores = compute_geometry_metrics(Path("gt.obj"), Path("pred.obj"))
    >>> scores["chamfer_distance"]          # tensor([0.0023])
    >>> scores["chamfer_distance"].mean()   # tensor(0.0023)

    >>> # Batched — per-item scores
    >>> scores = compute_geometry_metrics(
    ...     [Path("gt1.obj"), Path("gt2.obj")],
    ...     [Path("pred1.obj"), Path("pred2.obj")],
    ... )
    >>> scores["chamfer_distance"]          # tensor([0.0023, 0.0041])
    """
    if metrics is None:
        metrics = list(AVAILABLE_METRICS)

    # ─── Validate requested metric names ───
    unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. "
            f"Available: {list(AVAILABLE_METRICS)}"
        )

    # ─── Compute each metric and convert to tensor ───
    results: dict[str, torch.Tensor] = {}
    for name in metrics:
        fn = _METRIC_REGISTRY[name]
        score = fn(target, data, mode=mode, num_points=num_points, thresholds=thresholds)
        if isinstance(score, list):
            results[name] = torch.tensor(score)
        else:
            results[name] = torch.tensor([score])

    return results

