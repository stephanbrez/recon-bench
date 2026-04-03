import torch

from . import core
from .. import _types

# Registry mapping metric names to their core functions.
# Add new image metrics here — compute_image_metrics picks them up automatically.
_METRIC_REGISTRY: dict[str, callable] = {
    "psnr": core.psnr,
    "ssim": core.ssim,
    "ssim_windowed": core.ssim_windowed,
    "lpips": core.lpips,
}

AVAILABLE_METRICS = tuple(_METRIC_REGISTRY.keys())


def compute_image_metrics(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
    metrics: list[str] | None = None,
    shard_size: int = 10,
    max_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute image quality metrics between target and predicted images.

    Runs the requested metrics and returns per-item scores as tensors.
    Accepts single images or batched inputs in any supported format.

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s). Accepts pathlib.Path, PIL.Image, np.ndarray,
        torch.Tensor, or lists of these.
    data : ImageInput or list[ImageInput]
        Predicted image(s). Must match the batch size of target.
    metrics : list[str] or None
        Names of metrics to compute. None computes all available metrics:
        "psnr", "ssim", "ssim_windowed", "lpips".
    shard_size : int
        Maximum number of images per shard passed to each metric. Reduce to
        limit peak GPU memory. Default is 10.
    max_size : int or None
        If set, downscale images so the longest edge is at most this many
        pixels before computing metrics. Default is None (no resize).

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
    >>> scores = compute_image_metrics(Path("gt.png"), Path("pred.png"))
    >>> scores["psnr"]          # tensor([32.4])
    >>> scores["psnr"].mean()   # tensor(32.4)

    >>> # Batched — per-item scores
    >>> scores = compute_image_metrics(
    ...     [Path("gt1.png"), Path("gt2.png")],
    ...     [Path("pred1.png"), Path("pred2.png")],
    ... )
    >>> scores["psnr"]          # tensor([32.4, 28.1])
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

    # ─── Compute each metric ───
    results: dict[str, torch.Tensor] = {}
    for name in metrics:
        fn = _METRIC_REGISTRY[name]
        results[name] = fn(target, data, shard_size=shard_size, max_size=max_size)

    return results
