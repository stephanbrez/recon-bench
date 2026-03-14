"""
Open3D Camera Conversion
========================

This module converts the lightweight Camera dataclass into the two objects
that Open3D's OffscreenRenderer needs to position the camera in the scene:
an **intrinsic** matrix and an **extrinsic** matrix.

─── Concept: Intrinsic Matrix ───────────────────────────────────────────────

The intrinsic matrix K maps 3D camera-space points to 2D pixel coordinates.
For a pinhole camera it is:

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

Where:
  - fx, fy : focal lengths in pixels
  - cx, cy : principal point (image center, in pixels)

Given a vertical field-of-view angle (fov_y) and image height H:

    fy = (H / 2) / tan(fov_y / 2)
    fx = fy          # square pixels assumed
    cx = W / 2
    cy = H / 2

─── Concept: Extrinsic Matrix ───────────────────────────────────────────────

The extrinsic matrix is a 4x4 rigid-body transform that converts world-space
points into camera-space points. It encodes where the camera is and where it's
looking.

Open3D follows the OpenCV convention:
  - Camera looks down its +Z axis
  - X axis points right, Y axis points DOWN (not up)

To build the extrinsic from (position, look_at, up):

  1. Compute the forward vector (camera → look_at, normalized).
  2. Compute the right vector = normalize(forward × up).
  3. Recompute the true up = right × forward.
     (This orthogonalizes the up vector in case it wasn't exactly perpendicular.)
  4. Build a 3x3 rotation matrix R from these axes (rows in OpenCV convention):
       R = [right, true_up_flipped, forward]
       Note: In OpenCV convention the Y axis is flipped vs. the "math" convention.
  5. The translation t = -R @ position  (moves world origin to camera space).
  6. Assemble the 4x4 extrinsic:
       extrinsic = [[R[0,0], R[0,1], R[0,2], t[0]],
                    [R[1,0], R[1,1], R[1,2], t[1]],
                    [R[2,0], R[2,1], R[2,2], t[2]],
                    [0,      0,      0,      1    ]]

─── How OffscreenRenderer uses these ────────────────────────────────────────

    renderer.setup_camera(intrinsic, extrinsic, width, height)

The renderer internally constructs the projection matrix from the intrinsic
and uses the extrinsic to place the camera in world space.

─── Coordinate System Note ──────────────────────────────────────────────────

Open3D's 3D scene uses a right-handed coordinate system (X right, Y up, Z out
of screen toward viewer). The camera's extrinsic transforms from that system
into the OpenCV camera convention (X right, Y down, Z into screen). This Y-flip
is the main source of confusion — it's handled in step 4 above.

─── References ──────────────────────────────────────────────────────────────

- Open3D camera docs: http://www.open3d.org/docs/latest/tutorial/visualization/
- OpenCV pinhole model: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Look-at matrix derivation: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
"""

import numpy as np
import open3d as o3d

from .. import _types

def normalized_forward(
    camera: _types.Camera
) -> np.ndarray:
    """
    Returns the normalized forward direction of the camera.
    """
    forward = np.array(camera.look_at, dtype=np.float64) - np.array(camera.position, dtype=np.float64)
    return forward / np.linalg.norm(forward)

def camera_to_o3d_pinhole(
    camera: _types.Camera,
) -> tuple[o3d.camera.PinholeCameraIntrinsic, np.ndarray]:
    """
    Convert a Camera dataclass to Open3D pinhole intrinsic and extrinsic matrices.

    These are passed directly to OffscreenRenderer.setup_camera().

    Parameters
    ----------
    camera : Camera
        Camera specification with position, look_at, up, fov, width, height.

    Returns
    -------
    intrinsic : o3d.camera.PinholeCameraIntrinsic
        Pinhole camera intrinsic parameters (focal length, principal point).
    extrinsic : np.ndarray
        4x4 float64 extrinsic matrix (world → camera transform) in OpenCV
        convention (camera looks along +Z, Y points down).

    Notes
    -----
    See module docstring for the full derivation of each matrix.

    Raises
    ------
    ValueError
        If camera.up is parallel to the view direction (degenerate look-at).
    """

    # Get fy
    fy = (camera.height / 2) / np.tan(np.radians(camera.fov) / 2)
    fx = fy          # square pixels assumed
    cx = camera.width / 2
    cy = camera.height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=camera.width, height=camera.height, fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Look-at extrinsic
    # camera properties are tuples, so convert to arrays

    # create forward vector (normalized)
    position = np.array(camera.position, dtype=np.float64)
    forward = normalized_forward(camera)

    right = np.cross(forward, np.array(camera.up, dtype=np.float64))
    # Make sure right is orthogonal to forward
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError("camera.up is parallel to the view direction")
    right /= right_norm

    true_up = np.cross(forward, right)
    rot_mat = np.stack([right, -true_up, forward], axis=0) # (3, 3) float64 rotation matrix with Y down
    translation = -rot_mat @ position

    # could do this with: extrinsic = np.block([[rot_mat, translation[:, None]], [[0, 0, 0, 1]]])
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rot_mat
    extrinsic[:3, 3] = translation

    return (intrinsic, extrinsic)
