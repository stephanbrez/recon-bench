import pathlib
import numpy as np
import open3d as o3d
import open3d.t.geometry
import PIL.Image
import torch
import torchmetrics

from ..io import geometry
from ..io import image as _io_image
from ..utils import batch
from .. import _types

# Typing
from typing import Literal
from typing import Callable

# 📝 NOTE: DEVICE is defined here for GPU-accelerated image metrics (LPIPS).
# Open3D device is defined in io/geometry.py for geometry operations.
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ===== Image Metrics =====

def psnr(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
    shard_size: int = 10,
    max_size: int | None = None,
) -> torch.Tensor:
    """
    Calculate the Peak Signal to Noise Ratio (PSNR) for a batch of images.

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s). Single item or list of: pathlib.Path,
        PIL.Image, np.ndarray, or torch.Tensor.
        Tensor shapes: (N, C, H, W) or (C, H, W).
    data : ImageInput or list[ImageInput]
        Predicted image(s) to compare against the ground truth.
        Must match the batch size of target.
    shard_size : int
        Maximum number of images per shard. Reduce to limit peak GPU memory.
        Default is 10.
    max_size : int or None
        If set, downscale images so the longest edge is at most this many
        pixels before computing the metric. Default is None (no resize).

    Returns
    -------
    torch.Tensor
        PSNR scores, shape (N,). Higher is better.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    y_true = _to_rgb(_io_image.load_image(target, max_size))
    y_pred = _to_rgb(_io_image.load_image(data, max_size))
    _validate_image_batch(y_true, y_pred)

    return _sharded_calculate(_psnr_calc, y_true=y_true, y_pred=y_pred, shard_size=shard_size)

def _psnr_calc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Internal calculation for Peak Signal to Noise Ratio (PSNR).

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth image tensor of shape (N, C, H, W).
    y_pred : torch.Tensor
        Predicted image tensor of shape (N, C, H, W).

    Returns
    -------
    torch.Tensor
        PSNR scores, shape (N,).
    """
    return -10 * torch.log10((y_true - y_pred).pow(2).mean(dim=(1, 2, 3)))


def ssim(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
    shard_size: int = 10,
    max_size: int | None = None,
) -> torch.Tensor:
    """
    Calculate the Structural Similarity Index (SSIM) for a batch of images.
    Computes global SSIM (single mean/variance per image, not windowed).

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s). Single item or list of: pathlib.Path,
        PIL.Image, np.ndarray, or torch.Tensor.
        Tensor shapes: (N, C, H, W) or (C, H, W).
    data : ImageInput or list[ImageInput]
        Predicted image(s) to compare against the ground truth.
        Must match the batch size of target.
    shard_size : int
        Maximum number of images per shard. Reduce to limit peak GPU memory.
        Default is 10.
    max_size : int or None
        If set, downscale images so the longest edge is at most this many
        pixels before computing the metric. Default is None (no resize).

    Returns
    -------
    torch.Tensor
        SSIM scores, shape (N,). Range [-1, 1], higher is better.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    y_true = _to_rgb(_io_image.load_image(target, max_size))
    y_pred = _to_rgb(_io_image.load_image(data, max_size))
    _validate_image_batch(y_true, y_pred)

    return _sharded_calculate(_ssim_calc, y_true=y_true, y_pred=y_pred, shard_size=shard_size)

def _ssim_calc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Internal calculation for global Structural Similarity Index (SSIM).

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth image tensor of shape (N, C, H, W).
    y_pred : torch.Tensor
        Predicted image tensor of shape (N, C, H, W).

    Returns
    -------
    torch.Tensor
        SSIM scores, shape (N,).
    """

    # L=1.0 after normalization
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    target_mean = y_true.mean(dim=(1, 2, 3))
    data_mean = y_pred.mean(dim=(1, 2, 3))
    target_var = y_true.var(dim=(1, 2, 3))
    data_var = y_pred.var(dim=(1, 2, 3))
    sigma_td = (
        1 / (y_true.shape[1:].numel() - 1)  # C * H * W
        * (
            (y_true - target_mean[:, None, None, None])
            * (y_pred - data_mean[:, None, None, None])
        ).sum(dim=(1, 2, 3))
    )

    return (
        (2 * target_mean * data_mean + C1) * (2 * sigma_td + C2)
        / ((target_mean ** 2 + data_mean ** 2 + C1) * (target_var + data_var + C2))
    )

def ssim_windowed(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
    shard_size: int = 10,
    max_size: int | None = None,
) -> torch.Tensor:
    """
    Calculate SSIM using a sliding Gaussian window (torchmetrics implementation).
    For comparison against the global ssim() implementation.

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s). Single item or list of: pathlib.Path,
        PIL.Image, np.ndarray, or torch.Tensor.
        Tensor shapes: (N, C, H, W) or (C, H, W).
    data : ImageInput or list[ImageInput]
        Predicted image(s) to compare against the ground truth.
        Must match the batch size of target.
    shard_size : int
        Maximum number of images per shard. Reduce to limit peak GPU memory.
        Default is 10.
    max_size : int or None
        If set, downscale images so the longest edge is at most this many
        pixels before computing the metric. Default is None (no resize).

    Returns
    -------
    torch.Tensor
        SSIM scores, shape (N,). Range [-1, 1], higher is better.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    y_true = _to_rgb(_io_image.load_image(target, max_size))
    y_pred = _to_rgb(_io_image.load_image(data, max_size))
    _validate_image_batch(y_true, y_pred)

    metric = torchmetrics.image.StructuralSimilarityIndexMeasure(reduction="none")

    return _sharded_calculate(_ssim_windowed_calc,
        y_pred=y_pred,
        y_true=y_true,
        shard_size=shard_size,
        metric=metric,
    )

def _ssim_windowed_calc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Internal calculation for windowed SSIM using torchmetrics.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth image tensor of shape (N, C, H, W).
    y_pred : torch.Tensor
        Predicted image tensor of shape (N, C, H, W).
    **kwargs
        Additional keyword arguments to pass to the metric, including "metric".

    Returns
    -------
    torch.Tensor
        SSIM scores, shape (N,).
    """
    return kwargs["metric"](y_pred, y_true)

def lpips(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
    net: Literal["alex", "vgg", "squeeze"] = "alex",
    shard_size: int = 10,
    max_size: int | None = None,
) -> torch.Tensor:
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) for a
    batch of images using a pretrained feature network via torchmetrics.

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s). Single item or list of: pathlib.Path,
        PIL.Image, np.ndarray, or torch.Tensor.
        Tensor shapes: (N, C, H, W) or (C, H, W).
    data : ImageInput or list[ImageInput]
        Predicted image(s) to compare against the ground truth.
        Must match the batch size of target.
    net : {"alex", "vgg", "squeeze"}, optional
        Backbone network for feature extraction. "alex" (AlexNet) is the
        standard choice per the original paper. Defaults to "alex".
    shard_size : int
        Maximum number of images per shard. Reduce to limit peak GPU memory.
        Default is 10.
    max_size : int or None
        If set, downscale images so the longest edge is at most this many
        pixels before computing the metric. Default is None (no resize).

    Returns
    -------
    torch.Tensor
        LPIPS scores, shape (N,). Lower is more similar.

    Raises
    ------
    ValueError
        If net is not a valid backbone name, or if batch sizes differ.
    """
    VALID_NETS = ("alex", "vgg", "squeeze")
    if net not in VALID_NETS:
        raise ValueError(f"net must be one of {VALID_NETS}, got '{net}'")

    y_true = _to_rgb(_io_image.load_image(target, max_size)).to(DEVICE)
    y_pred = _to_rgb(_io_image.load_image(data, max_size)).to(DEVICE)
    _validate_image_batch(y_true, y_pred)

    # LPIPS expects [-1, 1]
    y_true = y_true * 2 - 1
    y_pred = y_pred * 2 - 1

    metric = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(
        net_type=net,
    ).to(DEVICE)

    return _sharded_calculate(
        _lpips_calc,
        y_true=y_true,
        y_pred=y_pred,
        shard_size=shard_size,
        metric=metric,
    )

def _lpips_calc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Internal calculation for LPIPS using torchmetrics.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth image tensor of shape (N, C, H, W).
    y_pred : torch.Tensor
        Predicted image tensor of shape (N, C, H, W).
    **kwargs
        Additional keyword arguments to pass to the metric, including "metric".

    Returns
    -------
    torch.Tensor
        LPIPS scores, shape (N,).
    """
    return torch.stack([
        kwargs["metric"](p.unsqueeze(0), t.unsqueeze(0))
        for p, t in zip(y_pred.unbind(0), y_true.unbind(0))
    ])
# ===== Geometry Metrics =====

def chamfer_distance(
    target: _types.MeshInput | list[_types.MeshInput],
    data: _types.MeshInput | list[_types.MeshInput],
    mode: _types.GeometryType = _types.GeometryType.MESH,
    num_points: int = 10000,
    **kwargs,
) -> float | list[float]:
    """
    Calculate chamfer distance between meshes or point clouds.

    Accepts single items or lists for batched evaluation.

    Parameters
    ----------
    target : MeshInput or list[MeshInput]
        Ground truth geometry/geometries. Each item is a pathlib.Path or
        GeometryArrays dict.
    data : MeshInput or list[MeshInput]
        Predicted geometry/geometries. Must match the batch size of target.
    mode : GeometryType
        Whether inputs are meshes or point clouds.
    num_points : int
        Number of points sampled from each surface for distance computation.

    Returns
    -------
    float or list[float]
        Chamfer distance per pair. Returns a scalar float when both inputs
        are single items, a list when both are lists.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    target_batch, was_single = batch.ensure_batch(target)
    data_batch, _ = batch.ensure_batch(data)
    batch.validate_batch_pair(target_batch, data_batch)

    load_fn = (
        geometry.load_mesh
        if mode == _types.GeometryType.MESH
        else geometry.load_point_cloud
    )
    dist_fn = (
        _chamfer_dist_meshes
        if mode == _types.GeometryType.MESH
        else _chamfer_dist_pointclouds
    )

    distances = [
        dist_fn(load_fn(t), load_fn(d), num_points)
        for t, d in zip(target_batch, data_batch)
    ]

    return batch.unbatch(distances, was_single)


# ===== Internal Single-Pair Implementations =====

def _chamfer_dist_meshes(
    mesh1: o3d.t.geometry.TriangleMesh,
    mesh2: o3d.t.geometry.TriangleMesh,
    num_points: int = 10000,
) -> float:
    """
    Compute chamfer distance between two Open3D triangle meshes.

    Parameters
    ----------
    mesh1 : o3d.t.geometry.TriangleMesh
    mesh2 : o3d.t.geometry.TriangleMesh
    num_points : int
        Surface sample count for distance computation.

    Returns
    -------
    float
    """
    params = o3d.t.geometry.MetricParameters(n_sampled_points=num_points)
    result = mesh1.compute_metrics(
        mesh2,
        (o3d.t.geometry.Metric.ChamferDistance,),
        params,
    )
    return float(result[0].cpu().numpy())


def _chamfer_dist_pointclouds(
    pcd1: o3d.t.geometry.PointCloud,
    pcd2: o3d.t.geometry.PointCloud,
    num_points: int = 10000,
) -> float:
    """
    Compute chamfer distance between two Open3D point clouds.

    Parameters
    ----------
    pcd1 : o3d.t.geometry.PointCloud
    pcd2 : o3d.t.geometry.PointCloud
    num_points : int
        Sample count for distance computation.

    Returns
    -------
    float
    """
    params = o3d.t.geometry.MetricParameters(n_sampled_points=num_points)
    result = pcd1.compute_metrics(
        pcd2,
        (o3d.t.geometry.Metric.ChamferDistance,),
        params,
    )
    return float(result[0].cpu().numpy())


def hausdorff_distance(
    target: _types.MeshInput | list[_types.MeshInput],
    data: _types.MeshInput | list[_types.MeshInput],
    mode: _types.GeometryType = _types.GeometryType.MESH,
    num_points: int = 10000,
    **kwargs,
) -> float | list[float]:
    """
    Calculate Hausdorff distance between meshes or point clouds.

    Accepts single items or lists for batched evaluation.

    Parameters
    ----------
    target : MeshInput or list[MeshInput]
        Ground truth geometry/geometries.
    data : MeshInput or list[MeshInput]
        Predicted geometry/geometries. Must match the batch size of target.
    mode : GeometryType
        Whether inputs are meshes or point clouds.
    num_points : int
        Number of points sampled from each surface for distance computation.

    Returns
    -------
    float or list[float]
        Hausdorff distance per pair.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    target_batch, was_single = batch.ensure_batch(target)
    data_batch, _ = batch.ensure_batch(data)
    batch.validate_batch_pair(target_batch, data_batch)

    load_fn = (
        geometry.load_mesh
        if mode == _types.GeometryType.MESH
        else geometry.load_point_cloud
    )
    dist_fn = (
        _hausdorff_dist_meshes
        if mode == _types.GeometryType.MESH
        else _hausdorff_dist_pointclouds
    )

    distances = [
        dist_fn(load_fn(t), load_fn(d), num_points)
        for t, d in zip(target_batch, data_batch)
    ]

    return batch.unbatch(distances, was_single)


def _hausdorff_dist_meshes(
    mesh1: o3d.t.geometry.TriangleMesh,
    mesh2: o3d.t.geometry.TriangleMesh,
    num_points: int = 10000,
) -> float:
    """
    Compute Hausdorff distance between two Open3D triangle meshes.

    Parameters
    ----------
    mesh1 : o3d.t.geometry.TriangleMesh
    mesh2 : o3d.t.geometry.TriangleMesh
    num_points : int
        Surface sample count for distance computation.

    Returns
    -------
    float
    """
    params = o3d.t.geometry.MetricParameters(n_sampled_points=num_points)
    result = mesh1.compute_metrics(
        mesh2,
        (o3d.t.geometry.Metric.HausdorffDistance,),
        params,
    )
    return float(result[0].cpu().numpy())


def _hausdorff_dist_pointclouds(
    pcd1: o3d.t.geometry.PointCloud,
    pcd2: o3d.t.geometry.PointCloud,
    num_points: int = 10000,
) -> float:
    """
    Compute Hausdorff distance between two Open3D point clouds.

    Parameters
    ----------
    pcd1 : o3d.t.geometry.PointCloud
    pcd2 : o3d.t.geometry.PointCloud
    num_points : int
        Sample count for distance computation.

    Returns
    -------
    float
    """
    params = o3d.t.geometry.MetricParameters(n_sampled_points=num_points)
    result = pcd1.compute_metrics(
        pcd2,
        (o3d.t.geometry.Metric.HausdorffDistance,),
        params,
    )
    return float(result[0].cpu().numpy())


def fscore(
    target: _types.MeshInput | list[_types.MeshInput],
    data: _types.MeshInput | list[_types.MeshInput],
    mode: _types.GeometryType = _types.GeometryType.MESH,
    num_points: int = 10000,
    thresholds: list[float] | None = None,
    **kwargs,
) -> list[float] | list[list[float]]:
    """
    Calculate F-score between meshes or point clouds at given thresholds.

    Accepts single items or lists for batched evaluation. Returns one F-score
    per threshold per pair.

    Parameters
    ----------
    target : MeshInput or list[MeshInput]
        Ground truth geometry/geometries.
    data : MeshInput or list[MeshInput]
        Predicted geometry/geometries. Must match the batch size of target.
    mode : GeometryType
        Whether inputs are meshes or point clouds.
    num_points : int
        Number of points sampled from each surface for distance computation.
    thresholds : list[float] or None
        F-score radii. Defaults to [0.01] if not provided.

    Returns
    -------
    list[float] or list[list[float]]
        F-scores per threshold for each pair. Returns a single list of floats
        (one per threshold) when both inputs are single items, a list of lists
        when both are lists.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    if thresholds is None:
        thresholds = [0.01]

    target_batch, was_single = batch.ensure_batch(target)
    data_batch, _ = batch.ensure_batch(data)
    batch.validate_batch_pair(target_batch, data_batch)

    load_fn = (
        geometry.load_mesh
        if mode == _types.GeometryType.MESH
        else geometry.load_point_cloud
    )
    score_fn = (
        _fscore_meshes
        if mode == _types.GeometryType.MESH
        else _fscore_pointclouds
    )

    scores = [
        score_fn(load_fn(t), load_fn(d), num_points, thresholds)
        for t, d in zip(target_batch, data_batch)
    ]

    return batch.unbatch(scores, was_single)


def _fscore_meshes(
    mesh1: o3d.t.geometry.TriangleMesh,
    mesh2: o3d.t.geometry.TriangleMesh,
    num_points: int,
    thresholds: list[float],
) -> list[float]:
    """
    Compute F-score between two Open3D triangle meshes.

    Parameters
    ----------
    mesh1 : o3d.t.geometry.TriangleMesh
    mesh2 : o3d.t.geometry.TriangleMesh
    num_points : int
        Surface sample count for distance computation.
    thresholds : list[float]
        F-score radii.

    Returns
    -------
    list[float]
        One F-score per threshold.
    """
    params = o3d.t.geometry.MetricParameters(
        n_sampled_points=num_points,
        fscore_radius=o3d.utility.DoubleVector(thresholds),
    )
    result = mesh1.compute_metrics(
        mesh2,
        (o3d.t.geometry.Metric.FScore,),
        params,
    )
    return [float(x) for x in result.cpu().numpy()]


def _fscore_pointclouds(
    pcd1: o3d.t.geometry.PointCloud,
    pcd2: o3d.t.geometry.PointCloud,
    num_points: int,
    thresholds: list[float],
) -> list[float]:
    """
    Compute F-score between two Open3D point clouds.

    Parameters
    ----------
    pcd1 : o3d.t.geometry.PointCloud
    pcd2 : o3d.t.geometry.PointCloud
    num_points : int
        Sample count for distance computation.
    thresholds : list[float]
        F-score radii.

    Returns
    -------
    list[float]
        One F-score per threshold.
    """
    params = o3d.t.geometry.MetricParameters(
        n_sampled_points=num_points,
        fscore_radius=o3d.utility.DoubleVector(thresholds),
    )
    result = pcd1.compute_metrics(
        pcd2,
        (o3d.t.geometry.Metric.FScore,),
        params,
    )
    return [float(x) for x in result.cpu().numpy()]


# ===== Internal Helpers =====

def _validate_image_batch(
    target: torch.Tensor,
    data: torch.Tensor,
) -> None:
    """
    Validate that two image tensors have matching batch sizes.

    Parameters
    ----------
    target : torch.Tensor
        Shape (N, C, H, W).
    data : torch.Tensor
        Shape (N, C, H, W).

    Raises
    ------
    ValueError
        If batch sizes (N) differ.
    """
    if target.shape[0] != data.shape[0]:
        raise ValueError(
            f"Batch size mismatch: target has {target.shape[0]} image(s), "
            f"data has {data.shape[0]} image(s)."
        )


def _to_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Drop the alpha channel from an image batch, returning RGB only.

    Perceptual metrics (PSNR/SSIM/LPIPS) are defined on RGB. Mixing alpha
    into them conflates pixel fidelity with mask fidelity. Inputs with 1
    or 3 channels are returned unchanged; 4-channel inputs are sliced.

    Parameters
    ----------
    image : torch.Tensor
        Shape (N, C, H, W).

    Returns
    -------
    torch.Tensor
        Shape (N, min(C, 3), H, W).
    """
    if image.shape[1] == 4:
        return image[:, :3]
    return image

def _sharded_calculate(
    calc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    shard_size: int = 10,
    **kwargs,
) -> torch.Tensor:
    """
    Apply a calculation function in shards to manage memory.

    Parameters
    ----------
    calc_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The metric calculation function to apply to each shard.
    y_true : torch.Tensor
        Ground truth tensor.
    y_pred : torch.Tensor
        Predicted tensor.
    shard_size : int
        Maximum number of samples per shard.
    **kwargs
        Additional keyword arguments to pass to `calc_fn`.

    Returns
    -------
    torch.Tensor
        The concatenated results of `calc_fn` across all shards.
    """

    if shard_size <= y_true.size(0):
        shards = []
        for shard_true, shard_pred in zip(
            torch.split(y_true, shard_size, dim=0),
            torch.split(y_pred, shard_size, dim=0),
        ):
            shards.append(calc_fn(shard_true, shard_pred, **kwargs))
        results = torch.cat(shards, dim=0)
    else:
        results = calc_fn(y_true, y_pred, **kwargs)
    return results
