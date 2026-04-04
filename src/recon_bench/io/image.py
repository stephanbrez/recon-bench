import pathlib

import numpy as np
import open3d as o3d
import PIL.Image
import torch

from ..utils import image as _utils_image
from .. import _types


def load_image(
    source: _types.ImageInput | list[_types.ImageInput],
    max_size: int | None = None,
) -> torch.Tensor:
    """
    Load one or more images from any supported source into a normalized tensor.

    Accepts a single item or a list for batched loading. All inputs are
    converted to float32 tensors with values in [0, 1].

    Parameters
    ----------
    source : ImageInput or list[ImageInput]
        One or more images as:
        - pathlib.Path : loaded via PIL (RGB)
        - PIL.Image.Image : converted directly
        - np.ndarray : (H,W,C), (C,H,W), or (N,C,H,W)
        - torch.Tensor : (C,H,W) or (N,C,H,W)
        When a list is provided, each element is loaded individually and
        the results are stacked into a single batch tensor.
    max_size : int or None
        If set, downscale so the longest edge is at most ``max_size`` pixels,
        preserving aspect ratio. Images already smaller are left unchanged.
        Only applied to path and PIL inputs; tensors and arrays are unchanged.

    Returns
    -------
    torch.Tensor
        Normalized tensor of shape (N, C, H, W) with values in [0, 1].
        N=1 for a single input.

    Raises
    ------
    TypeError
        If source is not a recognized ImageInput type.
    ValueError
        If list elements produce tensors with mismatched spatial dimensions.

    Examples
    --------
    >>> t = load_image(pathlib.Path("image.png"))
    >>> t.shape  # (1, 3, H, W)

    >>> batch = load_image([pathlib.Path("a.png"), pathlib.Path("b.png")])
    >>> batch.shape  # (2, 3, H, W)
    """
    # ─── Handle list: load each item, then stack into a batch ───
    if isinstance(source, list):
        tensors = [_load_single(s, max_size) for s in source]
        # torch.stack creates a new dim=0: [(1,C,H,W), ...] → (N,C,H,W)
        # squeeze the N=1 dim from each before stacking
        tensors = [t.squeeze(0) for t in tensors]
        return torch.stack(tensors, dim=0)

    return _load_single(source, max_size)


def _load_single(source: _types.ImageInput, max_size: int | None = None) -> torch.Tensor:
    """
    Load a single image into a normalized (1, C, H, W) tensor.

    Parameters
    ----------
    source : ImageInput
        A file path, PIL image, numpy array, or torch tensor.

    Returns
    -------
    torch.Tensor
        Shape (1, C, H, W), values in [0, 1].
    """
    # ─── Path: open as PIL RGB first ───
    if isinstance(source, pathlib.Path):
        source = PIL.Image.open(source).convert("RGB")

    if not isinstance(source, (PIL.Image.Image, np.ndarray, torch.Tensor)):
        raise TypeError(
            f"Unsupported ImageInput type: {type(source).__name__}. "
            "Expected pathlib.Path, PIL.Image, np.ndarray, or torch.Tensor."
        )

    # ─── Resize PIL images to max_size on the longest edge ───
    if max_size is not None and isinstance(source, PIL.Image.Image):
        w, h = source.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            source = source.resize(
                (round(w * scale), round(h * scale)),
                PIL.Image.LANCZOS,
            )

    return _utils_image.to_normalized_tensor(source)


def save_image(
    image: torch.Tensor | np.ndarray | PIL.Image.Image | o3d.geometry.Image,
    path: pathlib.Path,
) -> None:
    """
    Save an image to disk, inferring the format from the file suffix.

    Parameters
    ----------
    image : torch.Tensor, np.ndarray, PIL.Image.Image, or o3d.geometry.Image
        Image to save.
        - torch.Tensor : (C, H, W) or (1, C, H, W), values in [0, 1]
        - np.ndarray : (H, W, C), uint8 or float in [0, 1]
        - PIL.Image.Image : saved directly
        - o3d.geometry.Image : converted via numpy (H, W, 3) uint8
    path : pathlib.Path
        Destination file path. The suffix determines the format
        (e.g. ".png", ".jpg", ".bmp").

    Raises
    ------
    ValueError
        If a float tensor has values outside [0, 1].
    TypeError
        If image type is not supported.

    Examples
    --------
    >>> save_image(tensor, pathlib.Path("output/render.png"))
    """
    pil_image = _to_pil(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


def _to_pil(
    image: torch.Tensor | np.ndarray | PIL.Image.Image | o3d.geometry.Image,
) -> PIL.Image.Image:
    """Convert an image to PIL.Image.Image in uint8 RGB format."""
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")

    if isinstance(image, o3d.geometry.Image):
        image = np.asarray(image)

    if isinstance(image, torch.Tensor):
        # ─── Remove batch dim if present ───
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    f"save_image expects a single image, got batch of {image.shape[0]}."
                )
            image = image.squeeze(0)
        # (C, H, W) → (H, W, C)
        array = image.detach().cpu().permute(1, 2, 0).numpy()
    elif isinstance(image, np.ndarray):
        array = image
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[0] < array.shape[-1]:
            # Likely (C, H, W) — permute to (H, W, C)
            array = array.transpose(1, 2, 0)
    else:
        raise TypeError(
            f"Unsupported image type for saving: {type(image).__name__}."
        )

    # ─── Normalize float arrays to uint8 ───
    if array.dtype != np.uint8:
        if array.max() > 1.0 + 1e-6:
            raise ValueError(
                "Float image values must be in [0, 1] for saving."
            )
        array = (array * 255).clip(0, 255).astype(np.uint8)

    return PIL.Image.fromarray(array).convert("RGB")
