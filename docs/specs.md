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
│   ├── core.py              # Individual metric functions (psnr, ssim, lpips, chamfer)
│   ├── image.py             # Aggregator: compute all image metrics in one call
│   └── geometry.py          # Aggregator: compute all geometry metrics in one call
├── profiling/
│   ├── __init__.py          # Public exports: Timer, MemoryTracker, ProfileResult, etc.
│   ├── profile.py           # Result dataclasses: TimingEntry, MemoryEntry, ProfileResult
│   ├── timer.py             # Hierarchical wall-clock timer with CUDA sync
│   └── memory.py            # GPU memory tracker per section (torch.cuda)
├── utils/
│   ├── __init__.py
│   ├── image.py             # Tensor normalization (to_normalized_tensor)
│   ├── batch.py             # Batching helpers (ensure_batch, unbatch)
│   └── format.py            # Plain-text table and tree formatting
├── cli/
│   ├── __init__.py          # Top-level ``rb`` dispatcher (argparse subcommands)
│   └── eval_images.py       # ``rb eval-images``: batch image-vs-image evaluation
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
| `ProfileResult` | dataclass | `recon_bench.profiling.profile` |
| `TimingEntry` | dataclass | `recon_bench.profiling.profile` |
| `MemoryEntry` | dataclass | `recon_bench.profiling.profile` |
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
| `rendered_images` | `dict[str, torch.Tensor] \| None` | Renders keyed by "target"/"prediction"; `(C,H,W)` single view, `(N,C,H,W)` multi-view |
| `profile` | `ProfileResult \| None` | Timing and GPU memory data (when `profile=True`) |

Metric tensors are **not** mean-reduced — users call `.mean()`, `.std()`, or
index individual items as needed. This preserves per-view detail for multi-view
evaluation.

`summary()` returns a compact mean-metrics overview (one table per metric
group, plus the profiling tree if present). `detail(filenames=None)` returns a
per-item breakdown table when `N > 1`, using *filenames* as row labels (falls
back to numeric indices when omitted). The two methods are independent —
library users can call either or both.

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

Accepts `MeshInput | list[MeshInput]`. Supports mesh and point cloud modes
via `GeometryType`.

### Aggregators

- `compute_image_metrics(target, data, metrics=None)` → `dict[str, torch.Tensor]`
- `compute_geometry_metrics(target, data, metrics=None, ...)` → `dict[str, torch.Tensor]`

Both return per-item scores as tensors of shape `(N,)`. Users call `.mean()`
to reduce. Both use a registry pattern — new metrics are added to the
`_METRIC_REGISTRY` dict in the respective module.

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
4. For single-camera calls, `rendered_images` values are squeezed back to
   `(C, H, W)` for backward compatibility.

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

### Result Types (`profiling/profile.py`)

| Type | Fields | Description |
|---|---|---|
| `TimingEntry` | `name`, `duration_s`, `children` | Single timing node in the tree |
| `MemoryEntry` | `name`, `peak_mb`, `delta_mb`, `children` | Single memory node in the tree |
| `ProfileResult` | `timing`, `memory`, `cuda_available` | Aggregated report attached to `EvalResult` |

`ProfileResult.summary()` returns a human-readable tree with `├──`/`└──`
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
    from .profiling import profile as _profile_mod
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

| Package | Purpose |
|---|---|
| `torch` | Tensor operations, GPU compute |
| `numpy` | Array operations |
| `open3d` | Geometry I/O, mesh metrics, offscreen rendering |
| `torchmetrics` | SSIM (windowed), LPIPS implementations |
| `Pillow` | Image file I/O |
