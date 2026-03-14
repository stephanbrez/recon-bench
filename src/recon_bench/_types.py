from __future__ import annotations

import dataclasses
import enum
import math
import pathlib
import typing

import numpy as np
import PIL.Image
import torch

# Typing
from typing import TypedDict, NotRequired

if typing.TYPE_CHECKING:
    from .profiling import profile as _profile_mod

    ProfileResult = _profile_mod.ProfileResult
else:
    ProfileResult = typing.Any


# ===== Input Types =====

# Single image: tensor (C,H,W) or (N,C,H,W), numpy array, PIL image, or file path.
# Lists of these are accepted by all public functions for batched evaluation.
ImageInput = torch.Tensor | np.ndarray | PIL.Image.Image | pathlib.Path


class GeometryArrays(TypedDict):
    """
    Raw vertex/face arrays for in-memory mesh or point cloud construction.

    Fields
    ------
    verts : np.ndarray or torch.Tensor
        Vertex positions, shape (V, 3).
    faces : np.ndarray or torch.Tensor, optional
        Triangle face indices, shape (F, 3). Required for meshes.
    """
    verts: np.ndarray | torch.Tensor
    faces: NotRequired[np.ndarray | torch.Tensor]


# Single geometry item: file path or in-memory arrays.
# Lists of these are accepted by all public geometry functions for batched evaluation.
MeshInput = pathlib.Path | GeometryArrays


class GeometryType(enum.Enum):
    MESH = "mesh"
    POINTCLOUD = "pointcloud"


# ===== Camera =====

@dataclasses.dataclass(frozen=True, slots=True)
class Camera:
    """
    Lightweight, immutable camera specification for 3D rendering.

    Defines both the viewpoint (position/orientation) and the projection
    (field of view, image resolution, clipping planes). All angles in degrees.

    Parameters
    ----------
    position : tuple[float, float, float]
        Camera position in world coordinates (x, y, z).
    look_at : tuple[float, float, float]
        World-space point the camera is directed toward.
    up : tuple[float, float, float]
        World-space up vector. Must not be parallel to the view direction.
        Default is (0, 1, 0) — Y-up.
    fov : float
        Vertical field of view in degrees. Default 60.
    width : int
        Rendered image width in pixels. Default 512.
    height : int
        Rendered image height in pixels. Default 512.
    near : float
        Near clipping plane distance. Default 0.01.
    far : float
        Far clipping plane distance. Default 100.0.

    Examples
    --------
    >>> # Explicit camera
    >>> cam = Camera(position=(0, 0, 3), look_at=(0, 0, 0))

    >>> # Spherical orbit camera (most common for benchmarking)
    >>> cam = Camera.orbit(distance=2.5, elevation=30, azimuth=45)

    >>> # From a YAML config dict
    >>> cam = Camera.from_dict(yaml.safe_load(config_str))
    """
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 60.0
    width: int = 512
    height: int = 512
    near: float = 0.01
    far: float = 100.0

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        """
        Construct a Camera from a plain dictionary (e.g. loaded from YAML).

        Parameters
        ----------
        d : dict
            Dictionary with keys matching Camera field names. Tuple fields
            may be provided as lists (YAML sequences).

        Returns
        -------
        Camera
            Constructed camera instance.
        """
        fields = {f.name for f in dataclasses.fields(Camera)}
        kwargs = {k: tuple(v) if isinstance(v, list) else v
                  for k, v in d.items() if k in fields}
        return Camera(**kwargs)

    @staticmethod
    def orbit(
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 2.0,
        elevation: float = 30.0,
        azimuth: float = 0.0,
        **kwargs,
    ) -> "Camera":
        """
        Construct a camera positioned on a sphere orbiting around a center point.

        Useful for benchmarking where the subject is centered at the origin
        and you want a canonical viewpoint without computing position manually.

        Parameters
        ----------
        center : tuple[float, float, float]
            World-space point to orbit around (look_at target). Default origin.
        distance : float
            Radius of the orbit sphere. Default 2.0.
        elevation : float
            Elevation angle above the horizon in degrees. 0 = equator,
            90 = directly above. Default 30.
        azimuth : float
            Azimuth angle around the vertical axis in degrees. 0 = +Z axis,
            90 = +X axis. Default 0.
        **kwargs
            Additional Camera fields (fov, width, height, near, far, up).

        Returns
        -------
        Camera

        Examples
        --------
        >>> cam = Camera.orbit(distance=3.0, elevation=20, azimuth=135)
        """
        el_rad = math.radians(elevation)
        az_rad = math.radians(azimuth)
        x = center[0] + distance * math.cos(el_rad) * math.sin(az_rad)
        y = center[1] + distance * math.sin(el_rad)
        z = center[2] + distance * math.cos(el_rad) * math.cos(az_rad)
        return Camera(position=(x, y, z), look_at=center, **kwargs)

    @staticmethod
    def orbit_ring(
        num_views: int = 8,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 2.0,
        elevation: float = 30.0,
        **kwargs,
    ) -> list[Camera]:
        """
        Generate cameras evenly spaced around an orbit ring.

        Produces ``num_views`` cameras at equal azimuth intervals, all at
        the same elevation and distance. Useful for multi-view benchmarking.

        Parameters
        ----------
        num_views : int
            Number of cameras to generate. Default 8.
        center : tuple[float, float, float]
            World-space point to orbit around. Default origin.
        distance : float
            Radius of the orbit sphere. Default 2.0.
        elevation : float
            Elevation angle above the horizon in degrees. Default 30.
        **kwargs
            Additional Camera fields (fov, width, height, near, far, up).

        Returns
        -------
        list[Camera]
            ``num_views`` cameras spaced at azimuths
            ``[0, 360/num_views, 2*360/num_views, ...]``.

        Examples
        --------
        >>> cams = Camera.orbit_ring(num_views=4, distance=3.0)
        >>> len(cams)
        4
        """
        step = 360.0 / num_views
        return [
            Camera.orbit(
                center=center,
                distance=distance,
                elevation=elevation,
                azimuth=i * step,
                **kwargs,
            )
            for i in range(num_views)
        ]


# ===== Evaluation Result =====

@dataclasses.dataclass(slots=True)
class EvalResult:
    """
    Container for evaluation results returned by evaluate().

    Fields are None when that evaluation type was not performed.

    Parameters
    ----------
    image_metrics : dict[str, torch.Tensor] or None
        Image-quality scores keyed by metric name (e.g. "psnr", "ssim",
        "lpips"). Each value is a tensor of shape (N,) with one score per
        item or view. Call ``.mean()`` to reduce. None if no image
        evaluation was performed.
    geometry_metrics : dict[str, torch.Tensor] or None
        Geometry scores keyed by metric name (e.g. "chamfer_distance").
        Each value is a tensor of shape (N,). None if no geometry evaluation
        was performed.
    rendered_images : dict[str, torch.Tensor] or None
        Rendered images produced during evaluation, keyed by role:
        "target" and/or "prediction". Shape (C, H, W) for single view,
        (N, C, H, W) for multi-view. Values in [0, 1]. None if no
        rendering was performed.
    profile : ProfileResult or None
        Timing and GPU memory profiling data. Populated only when
        ``evaluate(..., profile=True)`` is used. None otherwise.
    """
    image_metrics: dict[str, torch.Tensor] | None = None
    geometry_metrics: dict[str, torch.Tensor] | None = None
    rendered_images: dict[str, torch.Tensor] | None = None
    profile: ProfileResult | None = None
