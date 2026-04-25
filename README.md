
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

### Per-Item Breakdown

`result.detail()` returns a per-item table. When targets were passed as
`Path` inputs, rows are labeled with those paths automatically; otherwise
items are numbered `[0], [1], ...`. Pass `filenames=...` to override.

```python
result = recon_bench.evaluate(
    [Path("gt/a.png"), Path("gt/b.png")],
    [Path("pr/a.png"), Path("pr/b.png")],
)
print(result.detail())  # rows labeled with target paths
```

### Recovering Targets

When targets come in as in-memory data (tensor / ndarray / PIL), the
loaded images are stashed on the result so you can save what was actually
evaluated:

```python
result = recon_bench.evaluate(target_tensor, pred_tensor)
result.target_images           # tensor (N, C, H, W), values in [0, 1]
result.save_targets("out/")    # writes target_0.png, target_1.png, ...
```

For Path inputs, `result.target_paths` holds the originals; `save_targets()`
is a no-op. Renders (when produced) are written by
`result.save_renders("out/")` as `prediction_{i}.png` /
`target_{i}.png` — same naming scheme.

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
| Hausdorff Distance | `"hausdorff_distance"` | Lower is better |
| F-score | `"fscore"` | Higher is better; one value per threshold |

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
Item              │    psnr  │    ssim  │  ssim_windowed  │   lpips
──┼──────────┼──────────┼──────────────────┼─────────
data/gt/img_00.png  │  32.410  │  0.9210  │         0.9050  │  0.0410
data/gt/img_01.png  │  28.130  │  0.8750  │         0.8520  │  0.0680
data/gt/img_02.png  │  31.200  │  0.9150  │         0.8980  │  0.0440
data/gt/img_03.png  │  29.118  │  0.8940  │         0.8785  │  0.0554
```

### `rb eval-pcd`

Evaluate predicted point clouds against a dense reference (e.g. COLMAP
`fused.ply`). Crops predictions to the reference bounding box, downsamples
to common voxel sizes, and reports Chamfer, Hausdorff, and F-score metrics.
Multiple predictions can be compared in a single run.

Assumes all clouds are already in the same coordinate frame. Use
`rb visualize-pcd` first to verify alignment.

| Short | Long | Description |
|---|---|---|
| `-r` | `--reference` | Reference point cloud path (required) |
| `-p` | `--pred` | Predictions as `name=path` pairs (required) |
| `-m` | `--metrics` | Metrics to compute; omit for all (`chamfer`, `hausdorff`, `fscore`) |
| `-v` | `--voxel-sizes` | Voxel sizes for downsampling (default: `0.01 0.02 0.05`) |
| `-f` | `--fscore-mults` | F-score radii as multiples of voxel size (default: `1.0 2.0 4.0`) |
| | `--crop-pad` | Padding added to reference bounding box before cropping |
| | `--remove-outliers` | Apply statistical outlier removal before downsampling |
| | `--write-csv` | Save results to CSV |
| | `--write-json` | Save results to JSON |

```bash
# Compare two methods against a COLMAP reference
rb eval-pcd -r fused.ply -p nerf=nerf.ply 3dgs=gs.ply

# Only chamfer and hausdorff, single voxel size
rb eval-pcd -r fused.ply -p nerf=nerf.ply -m chamfer hausdorff -v 0.02

# Full pipeline with outlier removal and CSV output
rb eval-pcd -r fused.ply -p nerf=nerf.ply 3dgs=gs.ply \
    --remove-outliers --write-csv results.csv
```

```
Loading reference: fused.ply
Reference points: 2847123
Reference AABB diagonal: 12.4832

Evaluating voxel_size=0.01, thresholds=[0.01, 0.02, 0.04]

=== 3dgs ===
voxel=0.01 | ref_points=84201 | pred_points=91432 | chamfer=0.00312 | hausdorff=0.04871
  F-score @ 0.01: 0.7823
  F-score @ 0.02: 0.9341
  F-score @ 0.04: 0.9887

=== nerf ===
voxel=0.01 | ref_points=84201 | pred_points=76554 | chamfer=0.00481 | hausdorff=0.07203
  F-score @ 0.01: 0.6914
  F-score @ 0.02: 0.8762
  F-score @ 0.04: 0.9541
```

### `rb visualize-pcd`

Render one or more point clouds to an image file using Open3D's offscreen
renderer. Works on headless servers — no display required. Useful for visually
verifying that predicted and reference clouds share the same coordinate frame
before running metrics.

Colors can be assigned per cloud or left to an automatic palette (blue, red,
green, ...).

| Short | Long | Description |
|---|---|---|
| | `clouds` | Point cloud paths, optionally with `:COLOR` suffix |
| `-o` | `--output` | Output image path (required) |
| `-W` | `--width` | Image width in pixels (default: `1920`) |
| `-H` | `--height` | Image height in pixels (default: `1080`) |
| | `--point-size` | Rendered point size (default: `2.0`) |
| | `--elevation` | Camera elevation in degrees (default: `25.0`) |
| | `--azimuth` | Camera azimuth in degrees (default: `45.0`) |
| | `--background` | Background color as `#RRGGBB` or `R,G,B` (default: `1,1,1` white) |

```bash
# Auto-colored overlay (blue, red, green, ...)
rb visualize-pcd ref.ply nerf.ply 3dgs.ply -o overlay.png

# Explicit per-cloud colors
rb visualize-pcd ref.ply:#3888E2 nerf.ply:#E34234 -o overlay.png

# Dark background, 4K resolution
rb visualize-pcd ref.ply pred.ply -o overlay.png \
    -W 3840 -H 2160 --background 0.1,0.1,0.1

# Adjust camera angle
rb visualize-pcd ref.ply pred.ply -o overlay.png --elevation 40 --azimuth 135
```

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Tensor operations, GPU compute |
| `numpy` | Array operations |
| `open3d` | Geometry I/O, mesh metrics, offscreen rendering |
| `torchmetrics` | SSIM (windowed), LPIPS implementations |
| `Pillow` | Image file I/O |
