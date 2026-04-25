"""
Open3D Offscreen Rendering
==========================

This module renders triangle meshes to image tensors using Open3D's
OffscreenRenderer — a headless GPU/CPU rasterizer that does not require
a display server.

─── Concept: OffscreenRenderer ──────────────────────────────────────────────

o3d.visualization.rendering.OffscreenRenderer(width, height) creates an
off-screen framebuffer of the given pixel dimensions. It maintains an internal
scene graph and a camera, and produces images when render_to_image() is called.

Typical lifecycle:

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)

    # 1. Configure scene
    renderer.scene.add_geometry("name", mesh_legacy, material)
    renderer.scene.set_background(color)    # RGBA floats in [0, 1]

    # 2. Set camera
    renderer.setup_camera(intrinsic, extrinsic, W, H)

    # 3. Render
    o3d_image = renderer.render_to_image()  # returns o3d.geometry.Image
    array = np.asarray(o3d_image)           # shape (H, W, 3), dtype uint8

─── Concept: Geometry Format ────────────────────────────────────────────────

OffscreenRenderer accepts legacy (non-tensor) Open3D geometry:
  - o3d.geometry.TriangleMesh  (NOT o3d.t.geometry.TriangleMesh)

To convert from tensor API:
    legacy_mesh = tensor_mesh.to_legacy()

─── Concept: MaterialRecord ─────────────────────────────────────────────────

Every geometry added to the scene needs a MaterialRecord that controls
its appearance:

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"   # physically-based lit surface

Common shader options:
  - "defaultLit"      : PBR lit surface (needs normals for correct shading)
  - "defaultUnlit"    : flat color, ignores lighting
  - "normals"         : visualizes surface normals as RGB (good for debugging)
  - "depth"           : visualizes depth buffer as grayscale

For clean evaluation renders, "defaultLit" or "defaultUnlit" are typical choices.
If the mesh has vertex normals, call mesh.to_legacy().compute_vertex_normals()
before adding to the scene.

─── Concept: Lighting ───────────────────────────────────────────────────────

The scene has an environment lighting system. Key methods:

    # Remove all existing lights
    renderer.scene.scene.enable_sun_light(False)

    # Add a directional (infinite) light
    renderer.scene.scene.add_directional_light(
        "sun",
        color=[1, 1, 1],          # RGB in [0, 1]
        direction=[0, -1, -0.5],  # points toward the light source
        intensity=75000,          # lux
        cast_shadows=True,
    )

    # Or use a preset environment (IBL — image-based lighting)
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS,
        [0, 0, 0],  # sun direction (ignored for IBL)
    )

For evaluation purposes, a single directional light co-located with the
camera is a common neutral choice:
    direction = normalize(camera.look_at - camera.position)

─── Concept: Headless Rendering on Linux ────────────────────────────────────

OffscreenRenderer requires one of:

  EGL (recommended for GPU):
    - Install: libegl1, libgl1-mesa-dev
    - Works with CUDA GPUs; Open3D auto-detects EGL if available.
    - Verify: python -c "import open3d; open3d.visualization.webrtc_server.enable_webrtc()"

  OSMesa (CPU fallback):
    - Install: libosmesa6-dev
    - Set environment variable before importing open3d:
        export OPEN3D_CPU_RENDERING=true
    - Or at runtime (must be before first o3d import):
        import os; os.environ["OPEN3D_CPU_RENDERING"] = "true"

If neither is available, OffscreenRenderer raises a RuntimeError at
instantiation time with a message about missing display or EGL.

─── References ──────────────────────────────────────────────────────────────

- Offscreen rendering guide:
  http://www.open3d.org/docs/latest/tutorial/visualization/headless_rendering.html
- MaterialRecord API:
  http://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.MaterialRecord.html
- Scene API:
  http://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.Open3DScene.html
"""

import numpy as np
import open3d as o3d
import torch

from .. import _types
from . import camera


def render_mesh(
    mesh: o3d.t.geometry.TriangleMesh,
    cam: _types.Camera,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Render a triangle mesh to an image tensor using Open3D's OffscreenRenderer.

    Parameters
    ----------
    mesh : o3d.t.geometry.TriangleMesh
        The mesh to render (tensor-API mesh, as returned by io.geometry.load_mesh).
        Vertex normals are computed automatically if not already present.
    cam : Camera
        Camera viewpoint and projection settings.
    background_color : tuple[float, float, float]
        RGB background color, each component in [0, 1]. Default white.

    Returns
    -------
    torch.Tensor
        Rendered image, shape (3, H, W), dtype float32, values in [0, 1].
        H and W match camera.height and camera.width.

    Raises
    ------
    RuntimeError
        If Open3D cannot initialize the OffscreenRenderer (missing EGL/OSMesa
        on headless Linux servers — see module docstring for setup instructions).

    Notes
    -----
    See module docstring for a full explanation of the OffscreenRenderer
    lifecycle, MaterialRecord options, and lighting setup.

    Examples
    --------
    >>> mesh = load_mesh(Path("model.obj"))
    >>> cam = Camera.orbit(distance=2.5, elevation=30, azimuth=45)
    >>> image = render_mesh(mesh, cam)
    >>> image.shape  # (3, 512, 512)
    """
    # ─── Step 1: Convert to legacy mesh ───
    legacy_mesh = mesh.to_legacy()
    if not legacy_mesh.has_vertex_normals():
        legacy_mesh.compute_vertex_normals()

    # ─── Step 2: Create renderer ───
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        cam.width, cam.height,
    )

    # Scene Config
    # ─── Step 3: Configure material ───
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    # ─── Step 4: Add geometry and background ───
    renderer.scene.add_geometry("mesh", legacy_mesh, material)
    renderer.scene.set_background(list(background_color) + [1.0])

    # ─── Step 5: Set up lighting ───
    renderer.scene.scene.enable_sun_light(False) # disable ambient
    direction = camera.normalized_forward(cam).tolist()
    renderer.scene.scene.add_directional_light(
        "sun",
        [1, 1, 1],
        direction,
        75000,
        cast_shadows=True,
    )

    # ─── Step 6: Set up camera ───
    intrinsics, extrinsics  = camera.camera_to_o3d_pinhole(cam)
    renderer.setup_camera(intrinsics, extrinsics)

    # ─── Step 7: Render ───
    o3d_image = renderer.render_to_image()  # (H, W, 3) uint8
    o3d_image = np.ascontiguousarray(np.flipud(np.asarray(o3d_image)))
    return torch.from_numpy(o3d_image).permute(2, 0, 1).float() / 255.0
