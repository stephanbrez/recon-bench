import contextlib
import pathlib

import numpy as np
import PIL.Image
import torch

from .io import image as _io_image
from .io import geometry as _io_geometry
from .metrics import image as _metrics_image
from .metrics import geometry as _metrics_geometry
from .rendering import renderer as _renderer
from .profiling import timer as _timer_mod
from .profiling import memory as _memory_mod
from .profiling import profile as _profile_mod
from . import _types

# Typing
from typing import Literal

EvalMode = Literal["image_vs_image", "image_vs_mesh", "mesh_vs_mesh"]

# File suffixes used for mode inference from pathlib.Path inputs.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".exr"}
_MESH_SUFFIXES = {".obj", ".ply", ".glb", ".gltf", ".off", ".stl", ".fbx"}


def evaluate(
    target: _types.ImageInput | _types.MeshInput | list[_types.ImageInput],
    prediction: _types.ImageInput | _types.MeshInput,
    mode: EvalMode | None = None,
    camera: _types.Camera | list[_types.Camera] | None = None,
    image_metrics: list[str] | None = None,
    image_eval: bool = False,
    geometry_type: _types.GeometryType = _types.GeometryType.MESH,
    num_points: int = 10000,
    profile: bool = False,
) -> _types.EvalResult:
    """
    Evaluate reconstruction quality between a target and a prediction.

    The evaluation mode is inferred automatically from the input types unless
    explicitly specified. Results are returned in an EvalResult container with
    optional image metrics, geometry metrics, and rendered images.

    Parameters
    ----------
    target : ImageInput, MeshInput, or list[ImageInput]
        The ground truth. Accepted types:
        - Image: pathlib.Path (.png/.jpg/etc), PIL.Image, np.ndarray, torch.Tensor
        - Geometry: pathlib.Path (.obj/.ply/etc) or GeometryArrays dict
        - list[ImageInput]: required for multi-view ``image_vs_mesh`` — one
          reference image per camera.
    prediction : ImageInput or MeshInput
        The prediction to evaluate. Same types as target.
    mode : {"image_vs_image", "image_vs_mesh", "mesh_vs_mesh"} or None
        Evaluation mode. Inferred from input types when None:
        - Both image-like → "image_vs_image"
        - Target image-like, prediction mesh-like → "image_vs_mesh"
        - Both mesh-like → "mesh_vs_mesh"
    camera : Camera, list[Camera], or None
        Camera(s) for rendering. A single Camera renders one view. A list
        of Cameras renders multiple views with per-view metrics.
        Required for "image_vs_mesh". For multi-view ``image_vs_mesh``,
        target must be a list of images matching the number of cameras.
        For "mesh_vs_mesh" with image_eval=True, defaults to
        Camera.orbit() if not provided.
        Use ``Camera.orbit_ring(num_views)`` to generate evenly spaced cameras.
    image_metrics : list[str] or None
        Which image metrics to compute. None means all: psnr, ssim,
        ssim_windowed, lpips.
    image_eval : bool
        For "mesh_vs_mesh" only: also render both meshes and compute image
        metrics. Default False.
    geometry_type : GeometryType
        For geometry evaluation: treat inputs as meshes or point clouds.
    num_points : int
        Surface sample count for chamfer distance computation.
    profile : bool
        If True, collect wall-clock timing and GPU memory usage for each
        evaluation step. Results are attached to ``EvalResult.profile``.
        Default False.

    Returns
    -------
    EvalResult
        Container with image_metrics, geometry_metrics, and/or rendered_images
        populated depending on the evaluation performed. Metric values are
        ``torch.Tensor`` of shape ``(N,)`` with one score per item/view.

    Raises
    ------
    ValueError
        If mode requires a camera but none is provided, if inputs are
        incompatible with the requested mode, or if multi-view camera count
        doesn't match target image count.
    TypeError
        If input types cannot be inferred as image or geometry.

    Examples
    --------
    >>> # Image vs image
    >>> result = evaluate(Path("gt.png"), Path("pred.png"))
    >>> result.image_metrics["psnr"]        # tensor([32.4])
    >>> result.image_metrics["psnr"].mean()  # tensor(32.4)

    >>> # Mesh vs mesh with chamfer distance
    >>> result = evaluate(Path("gt.obj"), Path("pred.obj"))
    >>> result.geometry_metrics["chamfer_distance"]  # tensor([0.0023])

    >>> # Multi-view mesh vs mesh evaluation
    >>> result = evaluate(
    ...     Path("gt.obj"), Path("pred.obj"),
    ...     image_eval=True,
    ...     camera=Camera.orbit_ring(num_views=8, distance=2.5),
    ... )
    >>> result.image_metrics["psnr"]  # tensor([...]) shape (8,)

    >>> # Multi-view image vs mesh (one reference image per camera)
    >>> cams = Camera.orbit_ring(num_views=4)
    >>> result = evaluate(
    ...     [Path("ref_0.png"), Path("ref_1.png"),
    ...      Path("ref_2.png"), Path("ref_3.png")],
    ...     Path("model.obj"),
    ...     camera=cams,
    ... )
    """
    # ─── Optional profiling setup ───
    timer = _timer_mod.Timer() if profile else None
    mem = _memory_mod.MemoryTracker() if profile else None

    # ─── Normalize camera to list ───
    cameras: list[_types.Camera] | None = None
    if isinstance(camera, _types.Camera):
        cameras = [camera]
    elif isinstance(camera, list):
        cameras = camera

    if mode is None:
        mode = _infer_mode(target, prediction)

    if mode == "image_vs_image":
        result = _eval_image_vs_image(
            target, prediction, image_metrics, timer, mem,
        )
    elif mode == "image_vs_mesh":
        result = _eval_image_vs_mesh(
            target, prediction, cameras, image_metrics, timer, mem,
        )
    elif mode == "mesh_vs_mesh":
        result = _eval_mesh_vs_mesh(
            target, prediction, cameras, image_metrics,
            image_eval, geometry_type, num_points, timer, mem,
        )
    else:
        raise ValueError(
            f"Unknown evaluation mode: '{mode}'. "
            "Expected 'image_vs_image', 'image_vs_mesh', or 'mesh_vs_mesh'."
        )

    # ─── Attach profiling report ───
    if profile:
        result.profile = _profile_mod.ProfileResult(
            timing=timer.get_report(),
            memory=mem.get_report(),
            cuda_available=mem.cuda_available,
        )

    return result


# ===== Profiling Helper =====

@contextlib.contextmanager
def _section(
    name: str,
    timer: _timer_mod.Timer | None,
    mem: _memory_mod.MemoryTracker | None,
):
    """
    Wrap a code block in optional timer and memory tracker sections.

    When both trackers are None (profiling disabled), this is a no-op.
    """
    timer_ctx = timer.section(name) if timer else contextlib.nullcontext()
    mem_ctx = mem.section(name) if mem else contextlib.nullcontext()
    with timer_ctx, mem_ctx:
        yield


# ===== Mode Handlers =====

def _eval_image_vs_image(
    target: _types.ImageInput,
    prediction: _types.ImageInput,
    image_metrics: list[str] | None,
    timer: _timer_mod.Timer | None,
    mem: _memory_mod.MemoryTracker | None,
) -> _types.EvalResult:
    """Compare two images directly."""
    with _section("compute_image_metrics", timer, mem):
        scores = _metrics_image.compute_image_metrics(
            target, prediction, image_metrics,
        )
    return _types.EvalResult(image_metrics=scores)


def _eval_image_vs_mesh(
    target: _types.ImageInput | list[_types.ImageInput],
    prediction: _types.MeshInput,
    cameras: list[_types.Camera] | None,
    image_metrics: list[str] | None,
    timer: _timer_mod.Timer | None,
    mem: _memory_mod.MemoryTracker | None,
) -> _types.EvalResult:
    """Render prediction mesh from each camera, compare to target image(s)."""
    if cameras is None:
        raise ValueError(
            "camera is required for 'image_vs_mesh' evaluation. "
            "Provide a Camera object or use Camera.orbit() to create one."
        )

    # ─── Validate multi-view: target list must match camera count ───
    if len(cameras) > 1:
        if not isinstance(target, list):
            raise ValueError(
                f"Multi-view image_vs_mesh requires a list of target images "
                f"matching the number of cameras ({len(cameras)}), "
                f"got a single target."
            )
        if len(target) != len(cameras):
            raise ValueError(
                f"Number of target images ({len(target)}) must match "
                f"number of cameras ({len(cameras)})."
            )

    with _section("load_prediction_mesh", timer, mem):
        mesh = _io_geometry.load_mesh(prediction)

    # ─── Render from each camera ───
    renders: list[torch.Tensor] = []
    with _section("render_prediction", timer, mem):
        for cam in cameras:
            renders.append(_renderer.render_mesh(mesh, cam))
    rendered_stack = torch.stack(renders)  # (N, C, H, W)

    with _section("load_target", timer, mem):
        target_tensor = _io_image.load_image(
            target if len(cameras) > 1 else target,
        )

    with _section("compute_image_metrics", timer, mem):
        scores = _metrics_image.compute_image_metrics(
            target_tensor, rendered_stack, image_metrics,
        )

    # ─── Squeeze renders for single-camera backward compat ───
    rendered_out = renders[0] if len(cameras) == 1 else rendered_stack

    return _types.EvalResult(
        image_metrics=scores,
        rendered_images={"prediction": rendered_out},
    )


def _eval_mesh_vs_mesh(
    target: _types.MeshInput,
    prediction: _types.MeshInput,
    cameras: list[_types.Camera] | None,
    image_metrics: list[str] | None,
    image_eval: bool,
    geometry_type: _types.GeometryType,
    num_points: int,
    timer: _timer_mod.Timer | None,
    mem: _memory_mod.MemoryTracker | None,
) -> _types.EvalResult:
    """Compute geometry metrics; optionally render both and compute image metrics."""
    with _section("compute_geometry_metrics", timer, mem):
        geo_scores = _metrics_geometry.compute_geometry_metrics(
            target, prediction, mode=geometry_type, num_points=num_points,
        )
    result = _types.EvalResult(geometry_metrics=geo_scores)

    if image_eval:
        # ─── Default camera if not provided ───
        if cameras is None:
            cameras = [_types.Camera.orbit(
                distance=2.0, elevation=30.0, azimuth=45.0,
            )]

        with _section("load_target_mesh", timer, mem):
            target_mesh = _io_geometry.load_mesh(target)

        with _section("load_prediction_mesh", timer, mem):
            pred_mesh = _io_geometry.load_mesh(prediction)

        # ─── Render from each camera ───
        target_renders: list[torch.Tensor] = []
        pred_renders: list[torch.Tensor] = []

        with _section("render_target", timer, mem):
            for cam in cameras:
                target_renders.append(_renderer.render_mesh(target_mesh, cam))

        with _section("render_prediction", timer, mem):
            for cam in cameras:
                pred_renders.append(_renderer.render_mesh(pred_mesh, cam))

        target_stack = torch.stack(target_renders)  # (N, C, H, W)
        pred_stack = torch.stack(pred_renders)      # (N, C, H, W)

        with _section("compute_image_metrics", timer, mem):
            img_scores = _metrics_image.compute_image_metrics(
                target_stack, pred_stack, image_metrics,
            )

        result.image_metrics = img_scores

        # ─── Squeeze for single-camera backward compat ───
        if len(cameras) == 1:
            result.rendered_images = {
                "target": target_renders[0],
                "prediction": pred_renders[0],
            }
        else:
            result.rendered_images = {
                "target": target_stack,
                "prediction": pred_stack,
            }

    return result


# ===== Mode Inference =====

def _infer_mode(
    target: _types.ImageInput | _types.MeshInput,
    prediction: _types.ImageInput | _types.MeshInput,
) -> EvalMode:
    """
    Infer the evaluation mode from the types of target and prediction.

    Parameters
    ----------
    target : ImageInput or MeshInput
    prediction : ImageInput or MeshInput

    Returns
    -------
    EvalMode

    Raises
    ------
    ValueError
        If inputs are incompatible (e.g. mesh target with image prediction).
    TypeError
        If a type cannot be classified.
    """
    target_is_image = _is_image(target)
    pred_is_image = _is_image(prediction)

    if target_is_image and pred_is_image:
        return "image_vs_image"
    if target_is_image and not pred_is_image:
        return "image_vs_mesh"
    if not target_is_image and not pred_is_image:
        return "mesh_vs_mesh"

    raise ValueError(
        "Cannot evaluate a mesh target against an image prediction. "
        "Swap target and prediction, or specify mode explicitly."
    )


def _is_image(value: _types.ImageInput | _types.MeshInput) -> bool:
    """
    Determine whether a value represents an image input.

    Parameters
    ----------
    value : ImageInput or MeshInput
        Input to classify.

    Returns
    -------
    bool
        True if value is image-like, False if geometry-like.

    Raises
    ------
    TypeError
        If the type is not recognized.
    """
    # ─── In-memory types are unambiguous ───
    if isinstance(value, (torch.Tensor, np.ndarray, PIL.Image.Image)):
        return True
    if isinstance(value, dict):  # GeometryArrays
        return False

    # ─── Paths: classify by file suffix ───
    if isinstance(value, pathlib.Path):
        suffix = value.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            return True
        if suffix in _MESH_SUFFIXES:
            return False
        raise TypeError(
            f"Cannot infer input type from path suffix '{suffix}'. "
            f"Known image suffixes: {sorted(_IMAGE_SUFFIXES)}. "
            f"Known mesh suffixes: {sorted(_MESH_SUFFIXES)}. "
            "Specify mode explicitly if using a non-standard extension."
        )

    # ─── Lists: classify from the first element ───
    if isinstance(value, list):
        if not value:
            raise ValueError("Cannot infer mode from an empty list.")
        return _is_image(value[0])

    raise TypeError(
        f"Unrecognized input type: {type(value).__name__}. "
        "Expected pathlib.Path, PIL.Image, np.ndarray, torch.Tensor, "
        "GeometryArrays dict, or a list of these."
    )
