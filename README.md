# recon-bench

A modular 3D reconstruction benchmarking toolkit. Evaluate reconstruction
quality across three modes — image vs image, image vs mesh, mesh vs mesh —
with a single function call.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from pathlib import Path
import recon_bench

# Image vs image — per-item tensor scores
result = recon_bench.evaluate(Path("gt.png"), Path("pred.png"))
result.image_metrics["psnr"]        # tensor([32.4])
result.image_metrics["psnr"].mean()  # tensor(32.4)

# Mesh vs mesh (chamfer distance)
result = recon_bench.evaluate(Path("gt.obj"), Path("pred.obj"))
result.geometry_metrics["chamfer_distance"]  # tensor([0.002])

# Mesh vs mesh with rendered image comparison
result = recon_bench.evaluate(
    Path("gt.obj"),
    Path("pred.obj"),
    image_eval=True,
    camera=recon_bench.Camera.orbit(distance=2.5, elevation=30),
)

# Render a mesh and compare to a reference image
cam = recon_bench.Camera(position=(0, 0, 3), look_at=(0, 0, 0))
result = recon_bench.evaluate(Path("reference.png"), Path("model.obj"), camera=cam)
```

All metric values are `torch.Tensor` of shape `(N,)` — one score per item or
view. Call `.mean()` to reduce to a scalar. Use `result.summary()` for a
formatted report:

```python
print(result.summary())
```

```
📊 Image Metrics
Metric  │  Mean
──┼──────
psnr    │  32.4000
ssim    │  0.9100
```

## Evaluation Modes

The evaluation mode is inferred automatically from input types, or set
explicitly via the `mode` parameter.

| Mode | Target | Prediction | Metrics |
|---|---|---|---|
| `image_vs_image` | Image | Image | PSNR, SSIM, SSIM (windowed), LPIPS |
| `image_vs_mesh` | Image | Mesh | Renders mesh, then image metrics |
| `mesh_vs_mesh` | Mesh | Mesh | Chamfer distance (+ optional image metrics) |

### Accepted Input Types

**Images**: `pathlib.Path` (`.png`, `.jpg`, etc.), `PIL.Image.Image`,
`np.ndarray`, `torch.Tensor`

**Geometry**: `pathlib.Path` (`.obj`, `.ply`, etc.),
`GeometryArrays` dict (`{"verts": ..., "faces": ...}`)

All functions accept single items or lists for batched evaluation.

## Camera

Use `Camera` to specify viewpoints for rendering.

```python
# Explicit position and target
cam = recon_bench.Camera(position=(0, 0, 3), look_at=(0, 0, 0))

# Spherical orbit (common for benchmarking)
cam = recon_bench.Camera.orbit(distance=2.5, elevation=30, azimuth=45)

# From a YAML config dict
cam = recon_bench.Camera.from_dict(config)
```

## Multi-View Evaluation

Generate cameras evenly spaced around an orbit and evaluate from all viewpoints:

```python
# 8 cameras at 45-degree azimuth intervals
cams = recon_bench.Camera.orbit_ring(num_views=8, distance=2.5, elevation=30)

# Mesh vs mesh — renders both meshes from all 8 views
result = recon_bench.evaluate(
    Path("gt.obj"), Path("pred.obj"),
    image_eval=True,
    camera=cams,
)
result.image_metrics["psnr"]  # tensor([...]) shape (8,) — per-view scores
result.image_metrics["psnr"].mean()   # mean across views
result.image_metrics["psnr"].argmin() # worst view
```

For **image vs mesh** with multiple cameras, you must provide one reference
image per camera — the camera list and image list must have the same length:

```python
cams = recon_bench.Camera.orbit_ring(num_views=4)
result = recon_bench.evaluate(
    [Path("ref_0.png"), Path("ref_1.png"),
     Path("ref_2.png"), Path("ref_3.png")],
    Path("model.obj"),
    camera=cams,
)
```

## Metrics

### Image Metrics

| Metric | Key | Range | Interpretation |
|---|---|---|---|
| PSNR | `"psnr"` | [0, +inf) | Higher is better |
| SSIM (global) | `"ssim"` | [-1, 1] | Higher is better |
| SSIM (windowed) | `"ssim_windowed"` | [-1, 1] | Higher is better |
| LPIPS | `"lpips"` | [0, +inf) | Lower is better |

### Geometry Metrics

| Metric | Key | Interpretation |
|---|---|---|
| Chamfer Distance | `"chamfer_distance"` | Lower is better |

Select specific metrics with the `image_metrics` parameter:

```python
result = recon_bench.evaluate(
    Path("gt.png"), Path("pred.png"),
    image_metrics=["psnr", "ssim"],
)
```

## Package-Level API

The package root re-exports the main entry points, so module-level imports are
supported and recommended for concise usage:

```python
import recon_bench

result = recon_bench.evaluate(Path("gt.png"), Path("pred.png"))
cam = recon_bench.Camera.orbit(distance=2.5, elevation=30)
timer = recon_bench.Timer()
mem = recon_bench.MemoryTracker()
```

For the full export list and source modules, see
`docs/specs.md` ("Public API" → "Package-Level Exports").

## Performance Profiling

Enable profiling to get wall-clock timing and GPU memory usage for each
evaluation step.

### Integrated with `evaluate()`

```python
result = recon_bench.evaluate(
    Path("gt.png"), Path("pred.png"),
    profile=True,
)
print(result.profile.summary())
```

```
⏱  Timing
└── compute_image_metrics: 1.8932s

🔋 GPU Memory
└── compute_image_metrics: peak 245.3 MiB, delta +12.1 MiB
```

### Standalone Usage

`Timer` and `MemoryTracker` can be used independently in your own code.

```python
import recon_bench

timer = recon_bench.Timer()
mem = recon_bench.MemoryTracker()

with timer.section("train"), mem.section("train"):
    with timer.section("forward"), mem.section("forward"):
        output = model(input_tensor)
    with timer.section("backward"), mem.section("backward"):
        loss.backward()

# Build a report
report = recon_bench.ProfileResult(
    timing=timer.get_report(),
    memory=mem.get_report(),
    cuda_available=mem.cuda_available,
)
print(report.summary())
```

```
⏱  Timing
└── train: 0.2451s
    ├── forward: 0.1023s
    └── backward: 0.1401s

🔋 GPU Memory
└── train: peak 512.0 MiB, delta +128.0 MiB
    ├── forward: peak 384.0 MiB, delta +256.0 MiB
    └── backward: peak 512.0 MiB, delta -128.0 MiB
```

Sections can be nested arbitrarily deep. On CPU-only systems, `MemoryTracker`
reports all zeros gracefully.

### Toggling Profiling

Pass `enabled=False` to either class to make all `section()` calls no-ops
without changing any call sites. `get_report()` returns an empty list when
disabled.

```python
profiling_on = False

timer = recon_bench.Timer(enabled=profiling_on)
mem = recon_bench.MemoryTracker(enabled=profiling_on)

with timer.section("train"):   # no-op when disabled
    train()

timer.get_report()  # [] when disabled
```

## CLI

Installing the package exposes the `rb` command with subcommands for
common evaluation tasks.

### `rb eval-images`

Batch image-vs-image evaluation. Images are sorted by name and paired
positionally. If directories have different image counts, a warning is
printed and evaluation proceeds up to the smaller count.

| Short | Long | Description |
|---|---|---|
| `-t` | `--target` | Directory of ground-truth images (required) |
| `-p` | `--prediction` | Directory of predicted images (required) |
| `-m` | `--metrics` | Space-separated metric names; omit for all |
| `-P` | `--profile` | Enable timing + GPU memory profiling |
| `-s` | `--summary-only` | Show only mean metrics, suppress per-item detail |
| `-M` | `--match-names` | Pair by filename stem instead of sorted position; unmatched files are skipped |

```bash
rb eval-images -t data/gt/ -p data/pred/
rb eval-images -t data/gt/ -p data/pred/ -m psnr ssim
rb eval-images -t data/gt/ -p data/pred/ -P
rb eval-images -t data/gt/ -p data/pred/ -s
```

```
Evaluating 4 image pairs ...

📊 Image Metrics
Metric          │    Mean
──┼──────────────────────
psnr            │  30.2145
ssim            │   0.9012
ssim_windowed   │   0.8834
lpips           │   0.0521

📊 Image Metrics (per item)
Item    │    psnr  │    ssim  │  ssim_windowed  │   lpips
──┼──────────┼──────────┼──────────────────┼─────────
img_00  │  32.410  │  0.9210  │         0.9050  │  0.0410
img_01  │  28.130  │  0.8750  │         0.8520  │  0.0680
img_02  │  31.200  │  0.9150  │         0.8980  │  0.0440
img_03  │  29.118  │  0.8940  │         0.8785  │  0.0554
```

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Tensor operations, GPU compute |
| `numpy` | Array operations |
| `open3d` | Geometry I/O, mesh metrics, offscreen rendering |
| `torchmetrics` | SSIM (windowed), LPIPS implementations |
| `Pillow` | Image file I/O |
