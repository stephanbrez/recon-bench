import torch
import PIL.Image
import numpy as np

DTYPE_DIVISORS = {
    torch.uint8:  255.0,
    torch.int16:  65535.0,
    torch.int32:  65535.0,
}


def _normalize_by_dtype(data: torch.Tensor) -> torch.Tensor:
    divisor = DTYPE_DIVISORS.get(data.dtype, None)
    if divisor is not None:
        return (data / divisor).float()
    return data.float()  # float32/float64 assumed already normalized


def to_normalized_tensor(img: np.ndarray | PIL.Image.Image | torch.Tensor) -> torch.Tensor:
    """
    Convert a PIL Image, numpy array, or tensor to a normalized tensor.

    Parameters
    ----------
    img : np.ndarray or PIL.Image.Image or torch.Tensor
        Input image to convert. Can be:
        - PIL Image: (H, W, C)
        - numpy array : (H, W, C) or (C, H, W) or (N, C, H, W)
        - torch.Tensor: (C, H, W) or (N, C, H, W)
    Returns
    -------
    torch.Tensor
        Normalized tensor with shape (N, C, H, W) and values in range [0, 1].
    """
    if not isinstance(img, torch.Tensor) and not isinstance(img, np.ndarray) and not isinstance(img, PIL.Image.Image):
        raise TypeError("input img must be of type torch.Tensor or numpy array or PIL.Image.Image")

    if isinstance(img, PIL.Image.Image):
        data = np.array(img)
        data = torch.from_numpy(data)               # (H, W, C)
        data = _normalize_by_dtype(data)
        data = data.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
    elif isinstance(img, np.ndarray):
        # get the shapes right
        data = torch.from_numpy(img)
        if data.ndim == 2:                          # (H, W) grayscale
            data = data.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        elif data.ndim == 3:
            if data.size(-1) == 1:                  # (H, W, 1) grayscale
                data = data.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
            elif data.size(-1) == 3:                # (H, W, C) color
                data = data.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            else:
                data = data.unsqueeze(0)            # (C, H, W) → (1, C, H, W)
        elif data.ndim == 4:
            if data.size(-1) == 3:                  # (N, H, W, C) color
                data = data.permute(0, 3, 1, 2)     # (N, C, H, W)
            # else already (N, C, H, W), no permute needed
        data = _normalize_by_dtype(data)
    elif isinstance(img, torch.Tensor):
        if img.ndim == 3:
            img = img.unsqueeze(0)                  # (C, H, W) → (N, C, H, W)
        data = _normalize_by_dtype(img)

    return data
