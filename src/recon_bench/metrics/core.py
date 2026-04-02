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

# 📝 NOTE: DEVICE is defined here for GPU-accelerated image metrics (LPIPS).
# Open3D device is defined in io/geometry.py for geometry operations.
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ===== Image Metrics =====

def psnr(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
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

    Returns
    -------
    torch.Tensor
        PSNR scores, shape (N,). Higher is better.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    y_true = _io_image.load_image(target)
    y_pred = _io_image.load_image(data)
    _validate_image_batch(y_true, y_pred)

    return -10 * torch.log10((y_true - y_pred).pow(2).mean(dim=(1, 2, 3)))


def ssim(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
) -> torch.Tensor:
    """
    Calculate the Structural Similarity Index (SSIM) for a batch of images.
    Computes global SSIM (single mean/variance per image, not windowed).

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s).
    data : ImageInput or list[ImageInput]
        Predicted image(s). Must match the batch size of target.

    Returns
    -------
    torch.Tensor
        SSIM scores, shape (N,). Range [-1, 1], higher is better.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    y_true = _io_image.load_image(target)
    y_pred = _io_image.load_image(data)
    _validate_image_batch(y_true, y_pred)

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
) -> torch.Tensor:
    """
    Calculate SSIM using a sliding Gaussian window (torchmetrics implementation).
    For comparison against the global ssim() implementation.

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s).
    data : ImageInput or list[ImageInput]
        Predicted image(s). Must match the batch size of target.

    Returns
    -------
    torch.Tensor
        SSIM scores, shape (N,). Range [-1, 1], higher is better.

    Raises
    ------
    ValueError
        If target and data batch sizes differ.
    """
    y_true = _io_image.load_image(target)
    y_pred = _io_image.load_image(data)
    _validate_image_batch(y_true, y_pred)

    metric = torchmetrics.image.StructuralSimilarityIndexMeasure(reduction="none")
    return metric(y_pred, y_true)


def lpips(
    target: _types.ImageInput | list[_types.ImageInput],
    data: _types.ImageInput | list[_types.ImageInput],
    net: Literal["alex", "vgg", "squeeze"] = "alex",
) -> torch.Tensor:
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) for a
    batch of images using a pretrained feature network via torchmetrics.

    Parameters
    ----------
    target : ImageInput or list[ImageInput]
        Ground truth image(s).
    data : ImageInput or list[ImageInput]
        Predicted image(s). Must match the batch size of target.
    net : {"alex", "vgg", "squeeze"}
        Backbone network for feature extraction. "alex" (AlexNet) is the
        standard choice per the original paper.

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

    y_true = _io_image.load_image(target).to(DEVICE)
    y_pred = _io_image.load_image(data).to(DEVICE)
    _validate_image_batch(y_true, y_pred)

    # LPIPS expects [-1, 1]
    y_true = y_true * 2 - 1
    y_pred = y_pred * 2 - 1

    metric = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(
        net_type=net
    ).to(DEVICE)
    return torch.stack([
        metric(p.unsqueeze(0), t.unsqueeze(0))
        for p, t in zip(y_pred.unbind(0), y_true.unbind(0))
    ])


# ===== Geometry Metrics =====

def chamfer_distance(
    target: _types.MeshInput | list[_types.MeshInput],
    data: _types.MeshInput | list[_types.MeshInput],
    mode: _types.GeometryType = _types.GeometryType.MESH,
    num_points: int = 10000,
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
