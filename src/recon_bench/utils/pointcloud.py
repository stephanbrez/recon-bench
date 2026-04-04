"""Point cloud preprocessing utilities for tensor-based Open3D point clouds."""
from __future__ import annotations

import dataclasses

import numpy as np
import open3d as o3d
import open3d.t.geometry


# ===== Data Structures =====

@dataclasses.dataclass(frozen=True, slots=True)
class AxisAlignedBoundingBox:
    """
    Axis-aligned bounding box (AABB) defined by min/max corner bounds.

    Edges are parallel to the coordinate axes. For oriented bounding boxes
    see ``open3d.t.geometry.OrientedBoundingBox``.

    Parameters
    ----------
    min_bound : np.ndarray
        Lower corner, shape (3,), float64.
    max_bound : np.ndarray
        Upper corner, shape (3,), float64.
    """

    min_bound: np.ndarray
    max_bound: np.ndarray


# ===== Functions =====

def compute_bounding_box(
    pcd: o3d.t.geometry.PointCloud,
    pad: float = 0.0,
) -> AxisAlignedBoundingBox:
    """
    Compute the axis-aligned bounding box of a point cloud.

    Parameters
    ----------
    pcd : o3d.t.geometry.PointCloud
        Input point cloud.
    pad : float
        Uniform padding added to each side of the bounding box.

    Returns
    -------
    AxisAlignedBoundingBox
        Bounding box with optional padding applied.
    """
    positions = pcd.point.positions.cpu().numpy().astype(np.float64)
    return AxisAlignedBoundingBox(
        min_bound=positions.min(axis=0) - pad,
        max_bound=positions.max(axis=0) + pad,
    )


def bounding_box_diagonal(bbox: AxisAlignedBoundingBox) -> float:
    """
    Compute the diagonal length of a bounding box.

    Parameters
    ----------
    bbox : AxisAlignedBoundingBox
        Bounding box.

    Returns
    -------
    float
        Euclidean length of the bounding box diagonal.
    """
    extent = bbox.max_bound - bbox.min_bound
    return float(np.linalg.norm(extent))


def crop_to_bounding_box(
    pcd: o3d.t.geometry.PointCloud,
    bbox: AxisAlignedBoundingBox,
) -> o3d.t.geometry.PointCloud:
    """
    Crop a point cloud to an axis-aligned bounding box.

    Parameters
    ----------
    pcd : o3d.t.geometry.PointCloud
        Input point cloud.
    bbox : AxisAlignedBoundingBox
        Bounding box to crop to.

    Returns
    -------
    o3d.t.geometry.PointCloud
        Cropped point cloud containing only positions within the box.

    Raises
    ------
    ValueError
        If cropping produces an empty point cloud.
    """
    positions = pcd.point.positions.cpu().numpy()
    mask = np.all(
        (positions >= bbox.min_bound) & (positions <= bbox.max_bound),
        axis=1,
    )
    if not mask.any():
        raise ValueError("Cropping produced an empty point cloud.")

    cropped = o3d.t.geometry.PointCloud()
    cropped.point.positions = o3d.core.Tensor(
        positions[mask], o3d.core.float32,
    )
    return cropped


def remove_outliers(
    pcd: o3d.t.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.t.geometry.PointCloud:
    """
    Remove statistical outliers from a point cloud.

    Parameters
    ----------
    pcd : o3d.t.geometry.PointCloud
        Input point cloud.
    nb_neighbors : int
        Number of neighbors for the statistical test.
    std_ratio : float
        Standard deviation ratio threshold.

    Returns
    -------
    o3d.t.geometry.PointCloud
        Cleaned point cloud with outliers removed.

    Raises
    ------
    ValueError
        If outlier removal produces an empty point cloud.
    """
    # 📝 NOTE: Statistical outlier removal is only available in the legacy API.
    legacy = pcd.to_legacy()
    cleaned, _ = legacy.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    if len(cleaned.points) == 0:
        raise ValueError("Outlier removal produced an empty point cloud.")
    return o3d.t.geometry.PointCloud.from_legacy(cleaned)


def voxel_downsample(
    pcd: o3d.t.geometry.PointCloud,
    voxel_size: float,
) -> o3d.t.geometry.PointCloud:
    """
    Downsample a point cloud using a voxel grid.

    Parameters
    ----------
    pcd : o3d.t.geometry.PointCloud
        Input point cloud.
    voxel_size : float
        Voxel edge length for downsampling.

    Returns
    -------
    o3d.t.geometry.PointCloud
        Downsampled point cloud.

    Raises
    ------
    ValueError
        If downsampling produces an empty point cloud.
    """
    out = pcd.voxel_down_sample(voxel_size)
    if out.is_empty():
        raise ValueError(
            f"Voxel downsampling produced an empty cloud at "
            f"voxel_size={voxel_size}"
        )
    return out


def num_points(pcd: o3d.t.geometry.PointCloud) -> int:
    """
    Return the number of points in a tensor point cloud.

    Parameters
    ----------
    pcd : o3d.t.geometry.PointCloud
        Input point cloud.

    Returns
    -------
    int
        Number of points.
    """
    return int(pcd.point.positions.shape[0])
