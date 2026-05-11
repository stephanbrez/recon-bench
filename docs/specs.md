# Recon-Bench Evaluation API — Specification

> **Scope**: This document is the internal developer reference — architecture,
> implementation patterns, type contracts, and extension points. It is written
> for contributors working on the library itself.
>
> For **user-facing documentation** (installation, API usage, CLI flags,
> examples), see [`README.md`](../README.md).

## Overview

Recon-bench is a modular 3D reconstruction benchmarking toolkit. It provides a
unified evaluation API so that external projects can assess reconstruction
quality with minimal setup — e.g. after completing a mesh fitting optimization,
call recon-bench to get PSNR, SSIM, LPIPS, and chamfer distance scores.

The API supports three evaluation modes:

- **Image vs Image** — compare a rendered/predicted image to a ground truth image
- **Image vs Mesh** — render a predicted mesh and compare it to a ground truth image
- **Mesh vs Mesh** — compute geometry metrics between two meshes, with optional
  image-based evaluation via rendering

All functions accept flexible input types (file paths, tensors, arrays, PIL
images) and handle both single-item and batched evaluation transparently.

## Module Structure

```
src/
├── __init__.py              # Public API exports
├── evaluate.py              # Orchestrator: top-level evaluate() function
├── _types.py                # Shared types: Camera, EvalResult, ImageInput, etc.
├── io/
│   ├── __init__.py
│   ├── image.py             # load_image, save_image
│   └── geometry.py          # load_mesh, save_mesh, load_point_cloud, save_point_cloud
├── rendering/
│   ├── __init__.py
│   ├── camera.py            # Camera → Open3D pinhole conversion
│   └── renderer.py          # Open3D OffscreenRenderer: mesh + camera → image tensor
├── metrics/
│   ├── __init__.py
│   ├── core.py              # Individual image and geometry metric functions
│   ├── image.py             # Aggregator: compute all image metrics in one call
│   └── geometry.py          # Aggregator: compute all geometry metrics in one call
├── profiling/
│   ├── __init__.py          # Public exports: Timer, MemoryTracker, ProfileResult, etc.
│   ├── _types.py            # Result dataclasses: TimingEntry, MemoryEntry, ProfileResult
│   ├── timer.py             # Hierarchical wall-clock timer with CUDA sync
│   └── memory.py            # GPU memory tracker per section (torch.cuda)
├── utils/
│   ├── __init__.py
│   ├── image.py             # Tensor normalization (to_normalized_tensor)
│   ├── batch.py             # Batching helpers (ensure_batch, unbatch)
│   └── format.py            # Plain-text table and tree formatting
├── cli/
│   ├── __init__.py          # Top-level ``rb`` dispatcher (argparse subcommands)
│   ├── eval_images.py       # ``rb eval-images``: batch image-vs-image evaluation
│   ├── eval_pcd.py          # ``rb eval-pcd``: point cloud evaluation
│   └── visualize_pcd.py     # ``rb visualize-pcd``: point cloud visualization
├── geometry/
├── datasets/
└── models/
```

**Structural conventions:**

- `pathlib.Path` for all file path parameters (no raw strings)
- Open3D dtypes used directly (`o3d.core.float32`, `o3d.core.int32`) — no aliases
- Device constants (`DEVICE`, `DEVICE_o3d`) defined locally in each module
- `utils/` contains only general-purpose helpers; domain-specific logic lives
  in its own package (`metrics/`, `io/`, `rendering/`)

## Public API

### Package-Level Exports (`import recon_bench`)

The package root (`recon_bench/__init__.py`) re-exports the primary API so
users can prefer module-level imports:

```python
import recon_bench
```

| Export | Kind | Source module |
|---|---|---|
| `evaluate` | function | `recon_bench.evaluate` |
| `Camera` | dataclass | `recon_bench._types` |
| `EvalResult` | dataclass | `recon_bench._types` |
| `GeometryArrays` | `TypedDict` | `recon_bench._types` |
| `GeometryType` | enum | `recon_bench._types` |
| `ProfileResult` | dataclass | `recon_bench.profiling._types` |
| `TimingEntry` | dataclass | `recon_bench.profiling._types` |
| `MemoryEntry` | dataclass | `recon_bench.profiling._types` |
| `Timer` | class | `recon_bench.profiling.timer` |
| `MemoryTracker` | class | `recon_bench.profiling.memory` |

`recon_bench.profiling` also exposes profiling-specific symbols directly for
users who prefer submodule imports.

### `evaluate()`

Top-level orchestrator that dispatches to the appropriate evaluation pipeline.

```python
evaluate(
    target: ImageInput | MeshInput | list[ImageInput],
    prediction: ImageInput | MeshInput,
    mode: EvalMode | None = None,
    camera: Camera | list[Camera] | None = None,
    image_metrics: list[str] | None = None,
    image_eval: bool = False,
    geometry_type: GeometryType = GeometryType.MESH,
    num_points: int = 10000,
    profile: bool = False,
    shard_size: int = 10,
    max_size: int | None = None,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> EvalResult
```

- `mode` is inferred from input types when `None`
- `camera` accepts a single `Camera` or `list[Camera]` for multi-view evaluation.
  Required for `image_vs_mesh`; defaults to `Camera.orbit()` for `mesh_vs_mesh`
  with `image_eval=True`
- For multi-view `image_vs_mesh`, `target` must be a `list[ImageInput]` of the
  same length as the camera list (one reference image per viewpoint)
- `image_metrics` selects which image metrics to compute; `None` means all
- `profile` enables wall-clock timing and GPU memory tracking per step;
  results are attached to `EvalResult.profile`
- `shard_size` limits how many images are processed per image-metric shard,
  reducing peak GPU memory for large batches
- `max_size` optionally downscales images so the longest edge is at most that
  many pixels before metric computation
- `background_color` controls rendered mesh background color as RGB floats in
  `[0, 1]`

**Mode inference** (when `mode=None`):

| Input type | Classification |
|---|---|
| `pathlib.Path` | Suffix-based: `.png`/`.jpg` → image, `.obj`/`.ply` → mesh |
| `PIL.Image.Image` | Image |
| `torch.Tensor` / `np.ndarray` | Image |
| `GeometryArrays` dict | Mesh |
| `list` | First element's type determines classification |

### Usage Examples

```python
from pathlib import Path
import recon_bench

# Image vs image — per-item tensor scores
result = recon_bench.evaluate(Path("gt.png"), Path("pred.png"))
result.image_metrics["psnr"]        # tensor([32.4])
result.image_metrics["psnr"].mean()  # tensor(32.4)

# Mesh vs mesh
result = recon_bench.evaluate(Path("gt.obj"), Path("pred.obj"))
result.geometry_metrics["chamfer_distance"]  # tensor([0.002])

# Mesh vs mesh + multi-view image evaluation
result = recon_bench.evaluate(
    Path("gt.obj"), Path("pred.obj"),
    image_eval=True,
    camera=recon_bench.Camera.orbit_ring(num_views=8, distance=2.5),
)
result.image_metrics["psnr"]  # tensor([...]) shape (8,) — per-view

# Multi-view image vs mesh (one reference image per camera)
cams = recon_bench.Camera.orbit_ring(num_views=4)
result = recon_bench.evaluate(
    [Path("ref_0.png"), Path("ref_1.png"),
     Path("ref_2.png"), Path("ref_3.png")],
    Path("model.obj"),
    camera=cams,
)
```

## Core Types

### Input Types

| Type | Accepted Formats |
|---|---|
| `ImageInput` | `pathlib.Path`, `PIL.Image.Image`, `np.ndarray`, `torch.Tensor` |
| `MeshInput` | `pathlib.Path`, `GeometryArrays` dict |

`GeometryArrays` is a `TypedDict` with `verts` (required, shape `(V, 3)`) and
`faces` (optional, shape `(F, 3)` — required for mesh operations).

### `Camera`

Frozen dataclass representing a camera viewpoint and projection.

| Field | Type | Default | Description |
|---|---|---|---|
| `position` | `tuple[float, float, float]` | — | World-space position |
| `look_at` | `tuple[float, float, float]` | — | Target point |
| `up` | `tuple[float, float, float]` | `(0, 1, 0)` | Up vector |
| `fov` | `float` | `60.0` | Vertical field of view (degrees) |
| `width` | `int` | `512` | Image width (px) |
| `height` | `int` | `512` | Image height (px) |
| `near` | `float` | `0.01` | Near clipping plane |
| `far` | `float` | `100.0` | Far clipping plane |

Factory methods:
- `Camera.orbit(center, distance, elevation, azimuth)` — spherical positioning
  around a target point
- `Camera.orbit_ring(num_views, center, distance, elevation)` — generate N
  cameras evenly spaced around an orbit at equal azimuth intervals
  (`i * 360 / num_views`). Returns `list[Camera]`.
- `Camera.from_dict(d)` — construct from a plain dict (e.g. loaded from YAML)

### `EvalResult`

Dataclass returned by `evaluate()`. Fields are `None` when that evaluation
type was not performed.

| Field | Type | Description |
|---|---|---|
| `image_metrics` | `dict[str, torch.Tensor] \| None` | Per-item image scores, each tensor shape `(N,)` |
| `geometry_metrics` | `dict[str, torch.Tensor] \| None` | Per-item geometry scores, each tensor shape `(N,)` |
| `rendered_images` | `dict[str, torch.Tensor] \| None` | Renders keyed by "target"/"prediction"; always `(N,C,H,W)`, including single-view output where `N == 1` |
| `target_paths` | `list[pathlib.Path] \| None` | Target image paths when path inputs were provided, used for per-item labels |
| `target_images` | `torch.Tensor \| None` | Loaded target images when in-memory image inputs were provided; `(N,C,H,W)` |
| `profile` | `ProfileResult \| None` | Timing and GPU memory data (when `profile=True`) |

Metric tensors are **not** mean-reduced — users call `.mean()`, `.std()`, or
index individual items as needed. This preserves per-view detail for multi-view
evaluation.

`summary()` returns a compact mean-metrics overview (one table per metric
group, plus the profiling tree if present). `detail(filenames=None)` returns a
per-item breakdown table when `N > 1`, using *filenames* as row labels (falls
back to numeric indices when omitted). The two methods are independent —
library users can call either or both.

`save_renders(output_dir)` writes any rendered images to disk as PNG files named
`{role}_{index}.png`. `save_targets(output_dir)` writes in-memory target images
as `target_{index}.png` when `target_images` is populated.

## Batching

Every public function handles both single and batched inputs. Users can call
any layer directly — metrics, I/O, or the orchestrator — without manually
wrapping inputs.

**Uniform `T | list[T]` interface:**

| Domain | Single | Batch | Return |
|---|---|---|---|
| Image | `ImageInput` → `(1,C,H,W)` | `list[ImageInput]` → `(N,C,H,W)` via `torch.stack` | `torch.Tensor (N,)` |
| Geometry | `MeshInput` → 1 item | `list[MeshInput]` → per-element | `torch.Tensor (N,)` |

**Batch size validation**: All paired-input functions (target & data) validate
matching batch sizes and raise `ValueError` on mismatch.

Batching helpers in `utils/batch.py`:
- `ensure_batch(inputs)` → `(list, was_single)` — normalizes to list form
- `unbatch(results, was_single)` → scalar or list — restores original shape
- `validate_batch_pair(target, data)` → raises on size mismatch

## Metrics

### Image Metrics

| Metric | Function | Range | Interpretation |
|---|---|---|---|
| PSNR | `psnr(target, data)` | [0, ∞) | Higher is better |
| SSIM (global) | `ssim(target, data)` | [-1, 1] | Higher is better |
| SSIM (windowed) | `ssim_windowed(target, data)` | [-1, 1] | Higher is better |
| LPIPS | `lpips(target, data, net)` | [0, ∞) | Lower is more similar |

All accept `ImageInput | list[ImageInput]` and return `torch.Tensor` shape `(N,)`.

### Geometry Metrics

| Metric | Function | Return | Interpretation |
|---|---|---|---|
| Chamfer Distance | `chamfer_distance(target, data, mode, num_points)` | `float \| list[float]` | Lower is better |
| Hausdorff Distance | `hausdorff_distance(target, data, mode, num_points)` | `float \| list[float]` | Lower is better |
| F-score | `fscore(target, data, mode, num_points, thresholds)` | `list[float] \| list[list[float]]` | Higher is better |

Accepts `MeshInput | list[MeshInput]`. Supports mesh and point cloud modes
via `GeometryType`.

### Aggregators

- `compute_image_metrics(target, data, metrics=None)` → `dict[str, torch.Tensor]`
- `compute_geometry_metrics(target, data, metrics=None, mode=GeometryType.MESH, num_points=10000, thresholds=None)` → `dict[str, torch.Tensor]`

Both return per-item scores as tensors of shape `(N,)`. Users call `.mean()`
to reduce. Both use a registry pattern — new metrics are added to the
`_METRIC_REGISTRY` dict in the respective module.

When multiple F-score thresholds are requested, geometry aggregation returns one
tensor per threshold using keys like `fscore_0.01`.

## I/O Layer

### Images (`io/image.py`)

| Function | Signature | Description |
|---|---|---|
| `load_image` | `(source: ImageInput \| list) → Tensor (N,C,H,W)` | Load and normalize to `[0, 1]` float32 |
| `save_image` | `(image, path: Path) → None` | Save to disk, format from suffix |

### Geometry (`io/geometry.py`)

| Function | Signature | Description |
|---|---|---|
| `load_mesh` | `(source: MeshInput) → TriangleMesh` | Load as Open3D tensor mesh |
| `save_mesh` | `(mesh, path: Path) → None` | Save to disk |
| `load_point_cloud` | `(source: MeshInput) → PointCloud` | Load as Open3D tensor point cloud |
| `save_point_cloud` | `(pcd, path: Path) → None` | Save to disk |

All save functions create parent directories automatically.

## Rendering

Uses Open3D's `OffscreenRenderer` for headless rasterization (no display
server required). Open3D was chosen over pytorch3d because differentiability
is not needed for evaluation and the API is simpler.

| Function | Signature | Description |
|---|---|---|
| `render_mesh` | `(mesh, camera, background_color) → Tensor (3,H,W)` | Render mesh from camera viewpoint |
| `camera_to_o3d_pinhole` | `(camera) → (Intrinsic, extrinsic_4x4)` | Convert Camera to Open3D format |

Headless Linux servers require EGL (GPU) or OSMesa (CPU fallback).

## Multi-View Evaluation

`evaluate()` accepts `camera: Camera | list[Camera] | None`. When a list is
provided, the prediction (and target, for mesh-vs-mesh) are rendered from each
viewpoint, producing per-view metric scores.

### How it works

1. The `camera` parameter is normalized to `list[Camera]` internally (a single
   `Camera` becomes a one-element list).
2. Mode handlers render from each camera in the list, stacking results into
   `(N, C, H, W)` tensors.
3. The stacked renders are passed to `compute_image_metrics`, which returns
   per-view `(N,)` tensors.
4. Single-camera calls keep the same `(N, C, H, W)` shape with `N == 1`.

### `Camera.orbit_ring()`

Generates cameras evenly spaced around an orbit:

```python
Camera.orbit_ring(
    num_views: int = 8,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance: float = 2.0,
    elevation: float = 30.0,
    **kwargs,
) -> list[Camera]
```

Azimuths are `i * 360 / num_views` for `i` in `range(num_views)`. Delegates
to `Camera.orbit()` for each camera.

### Mode-specific behavior

| Mode | Single Camera | Multi Camera |
|---|---|---|
| `image_vs_image` | No camera used | No camera used |
| `image_vs_mesh` | Render prediction, compare to target | Render from N views; target must be `list` of N images |
| `mesh_vs_mesh` + `image_eval` | Render both from 1 view | Render both from N views; per-view `(N,)` scores |

**⚠️ WARNING**: For multi-view `image_vs_mesh`, the target must be a
`list[ImageInput]` with one reference image per camera. The camera and image
lists must have the same length — each camera renders the mesh from one
viewpoint and compares to the corresponding reference image. A `ValueError` is
raised if the counts don't match.

## Profiling

The `profiling/` subpackage provides opt-in performance instrumentation for
wall-clock timing and GPU memory tracking. It is integrated into `evaluate()`
and also usable standalone.

### Architecture

Profiling uses two independent tracker classes (`Timer`, `MemoryTracker`) that
share the same section-based context manager interface. When `evaluate()` is
called with `profile=True`, it creates both trackers and wraps each evaluation
step via the `_section()` helper in `evaluate.py`, which composes both context
managers (or falls through as a no-op when profiling is disabled).

### `Timer` (`profiling/timer.py`)

Hierarchical wall-clock timer using `time.perf_counter()`.

```python
Timer(sync_cuda: bool = True, enabled: bool = True)
```

- `section(name)` — context manager; nestable for sub-steps
- `get_report()` → `list[TimingEntry]` — returns the timing tree

**CUDA synchronization**: When `sync_cuda=True` (default) and CUDA is
available, `torch.cuda.synchronize()` is called before each start/stop
measurement. This is required because GPU operations are asynchronous — without
synchronization, wall-clock timings will undercount GPU work.

**Toggling**: When `enabled=False`, `section()` is a no-op and `get_report()`
returns `[]`. CUDA synchronization is also skipped. This allows profiling to be
toggled via a flag without modifying call sites.

### `MemoryTracker` (`profiling/memory.py`)

GPU memory tracker using `torch.cuda` memory statistics.

```python
MemoryTracker(enabled: bool = True)
```

- `section(name)` — context manager; nestable for sub-steps
- `get_report()` → `list[MemoryEntry]` — returns the memory tree
- `cuda_available` — `bool` property

**Toggling**: When `enabled=False`, `section()` is a no-op and `get_report()`
returns `[]`. This allows profiling to be toggled via a flag without modifying
call sites.

**Key `torch.cuda` APIs used per section:**
- `reset_peak_memory_stats()` at entry
- `max_memory_allocated()` at exit → `peak_mb`
- `memory_allocated()` at entry/exit → `delta_mb`

**CPU fallback**: When CUDA is unavailable, all values report zero and
`cuda_available` is `False`. No errors are raised.

### Result Types (`profiling/_types.py`)

| Type | Fields | Description |
|---|---|---|
| `TimingEntry` | `name`, `duration_s`, `children` | Single timing node in the tree |
| `MemoryEntry` | `name`, `peak_mb`, `delta_mb`, `children` | Single memory node in the tree |
| `ProfileResult` | `timing`, `memory`, `cuda_available` | Aggregated report attached to `EvalResult` |

`ProfileResult.summary()` returns compact timing and memory tables.
`ProfileResult.detail()` returns a human-readable tree with `├──`/`└──`
connectors, displaying timing and memory sections hierarchically.

### Integration with `evaluate()`

The `_section()` helper in `evaluate.py` composes both trackers:

```python
@contextlib.contextmanager
def _section(name, timer, mem):
    timer_ctx = timer.section(name) if timer else contextlib.nullcontext()
    mem_ctx = mem.section(name) if mem else contextlib.nullcontext()
    with timer_ctx, mem_ctx:
        yield
```

Each mode handler (`_eval_image_vs_image`, `_eval_image_vs_mesh`,
`_eval_mesh_vs_mesh`) accepts optional `timer` and `mem` parameters and wraps
its steps (loading, rendering, metric computation) in `_section()` calls. When
profiling is disabled (`profile=False`), both are `None` and `_section()` is a
zero-cost no-op via `contextlib.nullcontext()`.

### Circular Import Avoidance

`_types.py` needs `ProfileResult` for the `EvalResult.profile` type annotation
but `profiling/` imports nothing from `_types.py`. The dependency is resolved
via `typing.TYPE_CHECKING`:

```python
if typing.TYPE_CHECKING:
    from .profiling import _types as _profile_mod
    ProfileResult = _profile_mod.ProfileResult
else:
    ProfileResult = typing.Any
```

This requires `from __future__ import annotations` at the top of `_types.py`
so that the annotation `ProfileResult | None` is a string at runtime (PEP 563).

## CLI

The package exposes a single `rb` entry point via `[project.scripts]` in
`pyproject.toml`, dispatched through `cli/__init__.py` using argparse
subcommands.

Current subcommands:

- `rb eval-images` — batch image-vs-image evaluation.
- `rb eval-pcd` — point cloud evaluation against a reference cloud.
- `rb visualize-pcd` — render a point cloud visualization.

### Adding a new subcommand

Each subcommand is a module in `cli/` that exposes two functions:

- `register(subparsers)` — adds the subparser and sets `func=run` as default
- `run(args)` — executes the command; heavy imports (e.g. `import recon_bench`)
  happen here to keep CLI startup fast

To add a command (e.g. `rb eval-meshes`):

1. Create `cli/eval_meshes.py` with `register()` and `run()`
2. Import and call `eval_meshes.register(subparsers)` in `cli/__init__.py`

### Conventions

- All flags must have both short (`-t`) and long (`--target`) forms
- Short flags use lowercase; `-P` (uppercase) is reserved for `--profile` to
  avoid conflict with `-p` (`--prediction`)

## FastAPI Service

The FastAPI service is an optional delivery surface over the existing library.
It lives inside the package at `src/recon_bench/service/`, not in a root-level
`app/` directory and not in a separate repository. The service must delegate to
the same evaluation code used by the Python API and CLI rather than duplicating
metric or rendering logic.

The service is additive. It must not change the public `evaluate()` contract,
CLI behavior, or package-level exports used by existing library consumers.

### Scope

The MVP service is upload-only. Clients submit image and geometry files in the
request; server-local filesystem paths remain a CLI/library concern.

Supported evaluation modes:

- `image-vs-image`
- `image-vs-mesh`
- `mesh-vs-mesh`

Service dependencies are optional so that normal library and CLI installations
stay lean. The intended run shape is:

```bash
uv run --extra service uvicorn recon_bench.service.asgi:app
```

### Package Structure

Initial service package layout:

```text
src/recon_bench/service/
├── __init__.py
├── asgi.py          # FastAPI app / create_app()
├── config.py        # Service settings: storage root, DB path, GPU slots, limits
├── routes.py        # /v1/evals, /v1/jobs, /v1/artifacts, /health
├── schemas.py       # Pydantic request/response models
├── storage.py       # Upload and artifact filesystem layout
├── serialization.py # EvalResult/Profile/tensor conversion to JSON-safe data
└── db.py            # SQLite persistence layer
```

This can be split into subpackages later if the service grows, but the MVP
should keep the surface small.

### Endpoints

MVP endpoints:

```text
POST /v1/evals/image-vs-image
POST /v1/evals/image-vs-mesh
POST /v1/evals/mesh-vs-mesh

GET /v1/jobs/{job_id}
GET /v1/jobs/{job_id}/result
GET /v1/artifacts/{artifact_id}
GET /health
```

The `POST /v1/evals/...` endpoints return a final JSON response after the
evaluation completes. Streaming is not required for the MVP.

### Request Models

FastAPI multipart uploads are represented as endpoint parameters, not fields on
the Pydantic request models. The Pydantic models describe the structured
options submitted alongside uploaded files.

```python
import datetime
import enum
import typing

import pydantic

import recon_bench


MetricName = typing.Annotated[str, pydantic.Field(min_length=4, max_length=100)]
RgbInt = typing.Annotated[int, pydantic.Field(ge=0, le=255)]
UnitVectorComponent = typing.Annotated[float, pydantic.Field(ge=-1.0, le=1.0)]


class EvalMode(enum.StrEnum):
    IMAGE_VS_IMAGE = "image-vs-image"
    IMAGE_VS_MESH = "image-vs-mesh"
    MESH_VS_MESH = "mesh-vs-mesh"


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SAVING_ARTIFACTS = "saving_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"


class CameraIn(pydantic.BaseModel):
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[UnitVectorComponent, UnitVectorComponent, UnitVectorComponent] = (
        0.0,
        1.0,
        0.0,
    )
    fov: float = pydantic.Field(default=60.0, gt=0.0, lt=180.0)
    width: int = pydantic.Field(default=512, gt=0, lt=10000)
    height: int = pydantic.Field(default=512, gt=0, lt=10000)
    near: float = pydantic.Field(default=0.01, gt=0.0)
    far: float = pydantic.Field(default=100.0, gt=0.0)

    @pydantic.model_validator(mode="after")
    def validate_planes(self) -> "CameraIn":
        if self.far <= self.near:
            raise ValueError("far must be greater than near")
        return self


CameraList = typing.Annotated[list[CameraIn], pydantic.Field(min_length=1)]


class CamerasPayload(pydantic.BaseModel):
    camera: CameraIn | None = None
    cameras: CameraList | None = None

    @pydantic.model_validator(mode="after")
    def validate_one_camera_source(self) -> "CamerasPayload":
        if self.camera is not None and self.cameras is not None:
            raise ValueError("provide either camera or cameras, not both")
        return self


class EvalOptions(pydantic.BaseModel):
    metrics: list[MetricName] | None = None
    profile: bool = False
    save_renders: bool = False
    shard_size: int = pydantic.Field(default=10, gt=0)
    max_size: int | None = pydantic.Field(default=None, gt=0)
    background_color: tuple[RgbInt, RgbInt, RgbInt] = (255, 255, 255)


class ImageVsImageOptions(EvalOptions):
    pass


class ImageVsMeshOptions(EvalOptions, CamerasPayload):
    @pydantic.model_validator(mode="after")
    def validate_camera_required(self) -> "ImageVsMeshOptions":
        if self.camera is None and self.cameras is None:
            raise ValueError("image-vs-mesh requires camera or cameras")
        return self


class MeshVsMeshOptions(EvalOptions, CamerasPayload):
    image_eval: bool = False
    geometry_type: recon_bench.GeometryType = recon_bench.GeometryType.MESH
    num_points: int = pydantic.Field(default=10000, gt=0, le=10000000)

    @pydantic.model_validator(mode="after")
    def validate_image_eval_geometry(self) -> "MeshVsMeshOptions":
        if self.image_eval and self.geometry_type == recon_bench.GeometryType.POINTCLOUD:
            raise ValueError("image_eval is only supported for mesh geometry")
        return self
```

`MetricName` constrains each string inside the `metrics` list. Applying
`min_length` or `max_length` directly to `list[str]` constrains the number of
items in the list, not the length of each metric name. Mode-specific endpoint
logic still validates requested metric names against the image or geometry
metric registries.

`background_color` accepts RGB integers in `[0, 255]`. The service normalizes
those values to floats in `[0, 1]` before calling `evaluate()`.

Endpoint file parameters remain separate from these models:

```python
target_files: list[UploadFile]
prediction_files: list[UploadFile]
options: ImageVsImageOptions | ImageVsMeshOptions | MeshVsMeshOptions
```

### Request Validation

The service validates request shape before calling `evaluate()` so user input
errors become `400` responses rather than internal server errors.

Upload safety rules:

- Never use client-provided filenames as filesystem paths.
- Generate server-side upload filenames under `runs/service/uploads/{job_id}/`.
- Store original filenames only as metadata for diagnostics.
- Reject unsupported suffixes using the same image and geometry suffix sets as
  the library.
- Enforce configurable upload size and file count limits before evaluation.

Mode-specific file rules:

- `image-vs-image` requires one or more target images and the same number of
  prediction images.
- `image-vs-mesh` requires one prediction mesh. With `camera`, it requires one
  target image. With `cameras`, it requires one target image per camera.
- `mesh-vs-mesh` MVP accepts one target geometry and one prediction geometry.
  Batched mesh-vs-mesh service requests are future work.

Camera rules:

- `image-vs-mesh` requires either `camera` or `cameras`.
- `mesh-vs-mesh` uses cameras only when `image_eval=True`.
- `image_eval=True` is supported only for `geometry_type=mesh`.

### Response Models

Metrics preserve the library behavior of returning per-item/per-view values
rather than only mean-reduced scores.

```python
class MetricValue(pydantic.BaseModel):
    name: str
    values: list[float]


class MetricsOut(pydantic.BaseModel):
    image: list[MetricValue] | None = None
    geometry: list[MetricValue] | None = None


class ArtifactOut(pydantic.BaseModel):
    artifact_id: str
    role: typing.Literal["target", "prediction"]
    index: int
    media_type: str
    url: str


class ProfileOut(pydantic.BaseModel):
    timing: list[dict[str, object]]
    memory: list[dict[str, object]]
    cuda_available: bool


class EvalResponse(pydantic.BaseModel):
    job_id: str
    status: JobStatus
    mode: EvalMode
    metrics: MetricsOut
    profile: ProfileOut | None = None
    artifacts: list[ArtifactOut] = pydantic.Field(default_factory=list)
    created_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class ErrorDetail(pydantic.BaseModel):
    code: str
    message: str
    details: dict[str, object] = pydantic.Field(default_factory=dict)
    job_id: str | None = None


class ErrorResponse(pydantic.BaseModel):
    error: ErrorDetail
```

### Persistence

Use SQLite for MVP persistence. The database records evaluation history, but it
does not act as a background job queue in the initial design.

The service initializes the database during FastAPI lifespan startup and uses
UUID string IDs for jobs and artifacts. A migration framework is unnecessary for
the MVP; schema evolution can be handled later if the service grows.

Persisted records:

- `jobs` — lifecycle status, mode, request configuration, timestamps, and error
  information
- `metrics` — individual metric values with item/view indexes
- `artifacts` — render metadata and filesystem-backed artifact locations

Job status values:

- `pending`
- `running`
- `saving_artifacts`
- `completed`
- `failed`

### Artifact Storage

Store uploads and generated artifacts under a service-specific runtime
directory using UUID-based job and artifact names:

```text
runs/service/
├── uploads/{job_id}/...
└── artifacts/{job_id}/{artifact_id}.png
```

Saved renders are returned as artifact records with URLs, not embedded directly
in the JSON response. The service should delegate image writing to the existing
I/O layer (`save_image`) rather than implementing a separate image writer.

### Concurrency Model

Concurrency is request/job-level, not intra-evaluation. FastAPI may accept and
prepare multiple requests concurrently, but the GPU-heavy evaluation section is
guarded by a configurable semaphore, defaulting to one concurrent evaluation.
Because `evaluate()` is synchronous and may block on CPU, GPU, and Open3D work,
the service runs it in a worker thread while the event loop remains responsive.

Lifecycle:

1. Receive request and upload files.
2. Create a `pending` job record.
3. Wait for the GPU semaphore.
4. Mark the job `running` and run `evaluate()` in a worker thread.
5. Convert metrics/results to JSON-safe CPU data.
6. Release the GPU semaphore.
7. Save render artifacts if requested.
8. Persist final metrics/artifacts and mark the job `completed`.
9. Return the final response.

This allows one request to upload and prepare while another request is running
evaluation. It also allows the next evaluation to start after the previous
request releases the GPU semaphore, even if the previous request is still doing
CPU/disk-side artifact work.

For MVP deployment, assume a single Uvicorn worker process. Cross-process or
multi-node GPU locking is future work.

### Error Handling

MVP evaluation is atomic: if one requested metric fails, the whole evaluation
fails. Partial metric success is deferred until the evaluator supports an
event-based or streaming flow.

HTTP error mapping:

- Request validation failure → `422`
- Unsupported file type or invalid mode options → `400`
- Missing uploaded file → `400`
- Upload too large → `413`
- Expected evaluator input errors (`ValueError`, `TypeError`) → mark job
  `failed`, return `400`
- Unexpected evaluation failure → mark job `failed`, return `500`
- Artifact save failure when `save_renders` was requested → mark job `failed`,
  return `500`
- Database failure → `500`

If failure happens before a job is created, return an HTTP error without a
`job_id`. If failure happens after job creation, persist the failed status and
include `job_id` in the error response.

Error responses should use a consistent shape:

```json
{
  "error": {
    "code": "evaluation_failed",
    "message": "Evaluation failed while computing metrics.",
    "details": {},
    "job_id": "..."
  }
}
```

### Streaming

Streaming is not part of the first MVP. If the final-response service lands
quickly, add NDJSON streaming as a bonus endpoint rather than SSE:

```text
POST /v1/evals/{mode}/stream
```

Initial stream events should be coarse lifecycle events:

- `accepted`
- `uploaded`
- `waiting_for_gpu`
- `evaluating`
- `saving_artifacts`
- `completed`
- `failed`

Metric-by-metric streaming requires refactoring the evaluation pipeline into an
event-emitting flow and is future work.

### Future Improvements

Deferred service features:

- Producer/consumer background job queue
- Server-Sent Events subscriptions
- Event replay
- Job cancellation
- Authentication and authorization
- Rate limits and quotas
- Artifact cleanup and retention policies
- Cross-process and multi-node GPU locking
- Remote/object storage for uploads and artifacts
- Server-local path inputs
- Metric-by-metric evaluator callbacks
- Partial metric success when one metric fails

## Design Decisions

| Decision | Rationale |
|---|---|
| `pathlib.Path` only (no `str`) | Modern Python convention; unambiguous type signatures; trivial mode inference from suffixes |
| Batching at every level | Users can call any layer directly without the orchestrator |
| Open3D for rendering | Simpler than pytorch3d for evaluation; no differentiability overhead |
| `Camera` as frozen dataclass | Immutable, hashable; `orbit()` factory covers the common benchmarking case |
| Metrics in `metrics/` not `utils/` | Domain-specific code belongs with its domain; `utils/` stays general-purpose |
| No DTYPE aliases | `o3d.core.float32` is explicit and self-documenting |
| Registry-based aggregators | New metrics are added by registering a single entry |
| Profiling as opt-in (`profile=False` default) | Zero overhead when not used; no import-time cost beyond the module |
| Separate `Timer` + `MemoryTracker` classes | Independently usable outside `evaluate()`; composable via `_section()` |
| CUDA sync in Timer | GPU ops are async; without sync, timings undercount GPU work |
| `TYPE_CHECKING` for `ProfileResult` in `_types.py` | Avoids circular import between `_types` ↔ `profiling` |
| Per-item tensor scores (no auto-mean) | Preserves per-view/per-item detail; users call `.mean()` when needed |
| `Camera.orbit_ring()` returns `list[Camera]` | Simple composition — no special multi-camera type; reuses `Camera.orbit()` |
| Multi-view `image_vs_mesh` requires matching `list[ImageInput]` | Each camera needs its own reference image; mismatched counts are a `ValueError` |
| Shared `utils/format.py` for display | Both `ProfileResult.summary()` and `EvalResult.summary()` use the same table/tree primitives; no external dep (e.g. rich) needed |
| `EvalResult.summary()` with optional filenames | Provides a ready-made report; filenames give human-friendly row labels without coupling to I/O |

## Dependencies

Core runtime dependencies:

| Package | Purpose |
|---|---|
| `torch` | Tensor operations, GPU compute |
| `numpy` | Array operations |
| `open3d` | Geometry I/O, mesh metrics, offscreen rendering |
| `torchmetrics` | SSIM (windowed), LPIPS implementations |
| `Pillow` | Image file I/O |

Optional service dependencies, installed via the `service` extra:

| Package | Purpose |
|---|---|
| `fastapi` | HTTP API framework and request validation |
| `uvicorn[standard]` | ASGI server for running the service |
| `pydantic` | Service request and response schemas |
| `aiosqlite` | Async SQLite access for service persistence |
| `python-multipart` | Multipart form/file upload parsing in FastAPI |
