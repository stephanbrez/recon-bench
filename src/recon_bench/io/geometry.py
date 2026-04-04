import pathlib

import numpy as np
import open3d as o3d
import open3d.t.geometry
import torch

from .. import _types

# 📝 NOTE: Device and dtype are defined locally; no aliases to keep things explicit.
_DEVICE = o3d.core.Device(
    "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
)


def load_mesh(
    source: "_types.GeometryArrays | pathlib.Path | o3d.t.geometry.TriangleMesh",
) -> o3d.t.geometry.TriangleMesh:
    """
    Load a triangle mesh from a file path, in-memory arrays, or pass through.

    Parameters
    ----------
    source : pathlib.Path, GeometryArrays, or o3d.t.geometry.TriangleMesh
        - pathlib.Path : file path to a mesh (e.g. .obj, .ply, .glb).
          Format is inferred from the suffix by Open3D.
        - GeometryArrays dict : must contain "verts" (V, 3) and "faces" (F, 3).
        - o3d.t.geometry.TriangleMesh : returned as-is (pass-through).

    Returns
    -------
    o3d.t.geometry.TriangleMesh
        Tensor-based triangle mesh on the default compute device.

    Raises
    ------
    ValueError
        If a GeometryArrays dict is missing the "faces" key.
    TypeError
        If source is not a supported type.

    Examples
    --------
    >>> mesh = load_mesh(pathlib.Path("model.obj"))
    >>> mesh = load_mesh({"verts": verts_array, "faces": faces_array})
    """
    if isinstance(source, o3d.t.geometry.TriangleMesh):
        return source

    if isinstance(source, pathlib.Path):
        if not source.exists():
            raise FileNotFoundError(f"No such file: '{source}'")
        return o3d.t.io.read_triangle_mesh(str(source))

    if isinstance(source, dict):
        if "faces" not in source:
            raise ValueError(
                "'faces' key is required in GeometryArrays for mesh loading."
            )
        mesh = o3d.t.geometry.TriangleMesh(_DEVICE)
        mesh.vertex.positions = _to_o3d_tensor(source["verts"], o3d.core.float32)
        mesh.triangle.indices = _to_o3d_tensor(source["faces"], o3d.core.int32)
        return mesh

    raise TypeError( # type: ignore
        f"Unsupported mesh source type: {type(source).__name__}. "
        "Expected pathlib.Path, GeometryArrays dict, or TriangleMesh."
    )


def save_mesh(mesh: o3d.t.geometry.TriangleMesh, path: pathlib.Path) -> None:
    """
    Save a triangle mesh to disk. Format inferred from the file suffix.

    Supported formats include .obj, .ply, .glb, .gltf (via Open3D).

    Parameters
    ----------
    mesh : o3d.t.geometry.TriangleMesh
        Mesh to save.
    path : pathlib.Path
        Destination file path.

    Examples
    --------
    >>> save_mesh(mesh, pathlib.Path("output/result.ply"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.t.io.write_triangle_mesh(str(path), mesh)


def load_point_cloud(
    source: "_types.GeometryArrays | pathlib.Path | o3d.t.geometry.PointCloud",
) -> o3d.t.geometry.PointCloud:
    """
    Load a point cloud from a file path, in-memory arrays, or pass through.

    Parameters
    ----------
    source : pathlib.Path, GeometryArrays, or o3d.t.geometry.PointCloud
        - pathlib.Path : file path to a point cloud (e.g. .ply, .pcd, .xyz).
          Format is inferred from the suffix by Open3D.
        - GeometryArrays dict : must contain "verts" (N, 3) with point positions.
          The "faces" key is ignored if present.
        - o3d.t.geometry.PointCloud : returned as-is (pass-through).

    Returns
    -------
    o3d.t.geometry.PointCloud
        Tensor-based point cloud on the default compute device.

    Raises
    ------
    TypeError
        If source is not a supported type.

    Examples
    --------
    >>> pcd = load_point_cloud(pathlib.Path("scan.ply"))
    >>> pcd = load_point_cloud({"verts": points_array})
    """
    if isinstance(source, o3d.t.geometry.PointCloud):
        return source

    if isinstance(source, pathlib.Path):
        if not source.exists():
            raise FileNotFoundError(f"No such file: '{source}'")
        pcd = o3d.t.io.read_point_cloud(
            str(source),
            remove_nan_points=True,
            remove_infinite_points=True,
        )
        if pcd.is_empty():
            raise ValueError(f"Loaded empty point cloud: {source}")
        return pcd

    if isinstance(source, dict):
        pcd = o3d.t.geometry.PointCloud(_DEVICE)
        pcd.point.positions = _to_o3d_tensor(source["verts"], o3d.core.float32)
        return pcd

    raise TypeError( # type: ignore
        f"Unsupported point cloud source type: {type(source).__name__}. "
        "Expected pathlib.Path, GeometryArrays dict, or PointCloud."
    )

def load_legacy_point_cloud(source: pathlib.Path) -> o3d.geometry.PointCloud:
    """
    Load a legacy point cloud from a file path.

    Parameters
    ----------
    source : pathlib.Path
        File path to a point cloud (e.g. .ply, .pcd, .xyz).
        Format is inferred from the suffix by Open3D.

    Returns
    -------
    o3d.geometry.PointCloud
        Legacy Open3D point cloud.

    Raises
    ------
    TypeError
        If source is not a supported type.

    Examples
    --------
    >>> pcd = load_legacy_point_cloud(pathlib.Path("scan.ply"))
    """
    if isinstance(source, pathlib.Path):
        if not source.exists():
            raise FileNotFoundError(f"No such file: '{source}'")
        pcd = o3d.io.read_point_cloud(
            source,
            remove_nan_points=True,
            remove_infinite_points=True,
        )
        if len(pcd.points) == 0:
            raise ValueError(f"Loaded empty point cloud: {source}")
        return pcd

    raise TypeError( # type: ignore
        f"Unsupported point cloud source type: {type(source).__name__}. "
        "Expected pathlib.Path or GeometryArrays dict."
    )


def save_point_cloud(
    pcd: o3d.t.geometry.PointCloud,
    path: pathlib.Path,
) -> None:
    """
    Save a point cloud to disk. Format inferred from the file suffix.

    Supported formats include .ply, .pcd, .xyz (via Open3D).

    Parameters
    ----------
    pcd : o3d.t.geometry.PointCloud
        Point cloud to save.
    path : pathlib.Path
        Destination file path.

    Examples
    --------
    >>> save_point_cloud(pcd, pathlib.Path("output/scan.ply"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.t.io.write_point_cloud(str(path), pcd)


# ===== Internal Helpers =====

def _to_o3d_tensor(
    data: np.ndarray | torch.Tensor,
    dtype: o3d.core.Dtype,
) -> o3d.core.Tensor:
    """
    Convert a numpy array or torch tensor to an Open3D core Tensor.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Source data.
    dtype : o3d.core.Dtype
        Target Open3D dtype (e.g. o3d.core.float32, o3d.core.int32).

    Returns
    -------
    o3d.core.Tensor
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return o3d.core.Tensor(data, dtype, _DEVICE)
