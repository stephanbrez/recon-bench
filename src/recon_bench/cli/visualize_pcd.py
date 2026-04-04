"""
``rb visualize-pcd`` — render point clouds to an image file (headless).

Loads one or more point clouds, assigns each a distinct color, and renders
them to a PNG using Open3D's OffscreenRenderer. Useful for visually
verifying coordinate-frame alignment before running metrics.

Colors can be specified per cloud or left to automatic assignment.
"""
from __future__ import annotations

import argparse
import pathlib
import sys


# ===== Constants =====

_DEFAULT_PALETTE = [
    (0.220, 0.557, 0.886),   # blue
    (0.890, 0.259, 0.204),   # red
    (0.302, 0.686, 0.290),   # green
    (0.988, 0.757, 0.027),   # yellow
    (0.608, 0.349, 0.714),   # purple
    (1.000, 0.498, 0.055),   # orange
    (0.420, 0.557, 0.137),   # olive
    (0.694, 0.349, 0.157),   # brown
]


# ===== Helpers =====

def _parse_color(color_str: str) -> tuple[float, float, float]:
    """
    Parse a color string into an RGB float tuple.

    Accepts either a hex string (``#RRGGBB``) or three comma-separated
    floats (``R,G,B`` in [0, 1]).

    Parameters
    ----------
    color_str : str
        Color specification.

    Returns
    -------
    tuple[float, float, float]
        RGB values in [0, 1].

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    color_str = color_str.strip()

    if color_str.startswith("#") and len(color_str) == 7:
        r = int(color_str[1:3], 16) / 255.0
        g = int(color_str[3:5], 16) / 255.0
        b = int(color_str[5:7], 16) / 255.0
        return (r, g, b)

    parts = color_str.split(",")
    if len(parts) == 3:
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    raise ValueError(
        f"Cannot parse color '{color_str}'. "
        "Expected '#RRGGBB' or 'R,G,B' (floats in [0,1])."
    )


def _parse_cloud_specs(
    specs: list[str],
) -> list[tuple[pathlib.Path, str | None]]:
    """
    Parse cloud specs from the CLI.

    Each spec is either a bare path or ``path:color``.

    Parameters
    ----------
    specs : list[str]
        Raw CLI arguments.

    Returns
    -------
    list[tuple[pathlib.Path, str | None]]
        List of (path, color_string_or_None).
    """
    result: list[tuple[pathlib.Path, str | None]] = []
    for spec in specs:
        if ":" in spec:
            path_str, color_str = spec.rsplit(":", 1)
            result.append((pathlib.Path(path_str), color_str))
        else:
            result.append((pathlib.Path(spec), None))
    return result


# ===== CLI Registration =====

def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``visualize-pcd`` subcommand."""
    parser = subparsers.add_parser(
        "visualize-pcd",
        help="Render point clouds to an image file (headless).",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "clouds", nargs="+",
        help=(
            "Point cloud files to render. Optionally append "
            "':COLOR' to set a color per cloud, e.g. "
            "'ref.ply:#3888E2 pred.ply:#E34234'. "
            "COLOR is '#RRGGBB' or 'R,G,B' (floats). "
            "Uncolored clouds get automatic palette colors."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, required=True,
        help="Output image path (e.g. overlay.png).",
    )
    parser.add_argument(
        "-W", "--width", type=int, default=1920,
        help="Image width in pixels (default: 1920).",
    )
    parser.add_argument(
        "-H", "--height", type=int, default=1080,
        help="Image height in pixels (default: 1080).",
    )
    parser.add_argument(
        "--point-size", type=float, default=2.0,
        help="Rendered point size (default: 2.0).",
    )
    parser.add_argument(
        "--elevation", type=float, default=25.0,
        help="Camera elevation angle in degrees (default: 25.0).",
    )
    parser.add_argument(
        "--azimuth", type=float, default=45.0,
        help="Camera azimuth angle in degrees (default: 45.0).",
    )
    parser.add_argument(
        "--background", default="1,1,1",
        help=(
            "Background color as '#RRGGBB' or 'R,G,B' "
            "(default: '1,1,1' white)."
        ),
    )
    parser.set_defaults(func=run)


# ===== Subcommand Entry Point =====

def run(args: argparse.Namespace) -> None:
    """Execute the ``visualize-pcd`` subcommand."""
    import numpy as np
    import open3d as o3d
    import open3d.visualization.rendering

    from ..io import geometry

    cloud_specs = _parse_cloud_specs(args.clouds)
    background = _parse_color(args.background)

    # ─── Load and color point clouds ───
    legacy_clouds: list[o3d.geometry.PointCloud] = []
    palette_idx = 0

    for cloud_path, color_str in cloud_specs:
        if not cloud_path.exists():
            print(f"Error: file not found: {cloud_path}")
            sys.exit(1)

        print(f"Loading: {cloud_path}")
        pcd_tensor = geometry.load_point_cloud(cloud_path)
        pcd = pcd_tensor.to_legacy()

        if color_str is not None:
            color = _parse_color(color_str)
        else:
            color = _DEFAULT_PALETTE[palette_idx % len(_DEFAULT_PALETTE)]
            palette_idx += 1

        pcd.paint_uniform_color(list(color))
        legacy_clouds.append(pcd)

    # ─── Compute combined bounding box for camera placement ───
    all_points = np.concatenate(
        [np.asarray(pcd.points) for pcd in legacy_clouds], axis=0,
    )
    center = all_points.mean(axis=0)
    extent = all_points.max(axis=0) - all_points.min(axis=0)
    diagonal = float(np.linalg.norm(extent))

    # ─── Set up offscreen renderer ───
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        args.width, args.height,
    )

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = args.point_size

    for i, pcd in enumerate(legacy_clouds):
        renderer.scene.add_geometry(f"cloud_{i}", pcd, material)

    renderer.scene.set_background(list(background) + [1.0])

    # ─── Position camera to see all clouds ───
    # Place camera at 1.5x the scene diagonal, looking at the center,
    # slightly elevated for a 3/4 view.
    distance = diagonal * 1.5
    elevation = np.radians(args.elevation)
    azimuth = np.radians(args.azimuth)
    cam_pos = center + distance * np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.cos(azimuth),
    ])

    renderer.setup_camera(
        60.0,
        o3d.geometry.AxisAlignedBoundingBox(
            all_points.min(axis=0), all_points.max(axis=0),
        ),
        center,
    )
    renderer.scene.camera.look_at(
        center.tolist(), cam_pos.tolist(), [0, 1, 0],
    )

    # ─── Render and save ───
    o3d_image = renderer.render_to_image()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(output_path), o3d_image)
    print(f"Saved: {output_path} ({args.width}x{args.height})")
