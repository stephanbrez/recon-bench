from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ensure_batch(inputs: T | list[T]) -> tuple[list[T], bool]:
    """
    Normalize a single item or list into a list, tracking whether the
    original input was a single item.

    Parameters
    ----------
    inputs : T or list[T]
        A single item or a list of items.

    Returns
    -------
    tuple[list[T], bool]
        A tuple of (batched_list, was_single). Pass was_single to unbatch()
        to restore the original return shape.

    Examples
    --------
    >>> items, was_single = ensure_batch(Path("a.obj"))
    >>> items
    [PosixPath('a.obj')]
    >>> was_single
    True

    >>> items, was_single = ensure_batch([Path("a.obj"), Path("b.obj")])
    >>> was_single
    False
    """
    if isinstance(inputs, list):
        return inputs, False
    return [inputs], True


def unbatch(results: list[R], was_single: bool) -> R | list[R]:
    """
    Return a scalar if the original input was a single item, or the full
    list if it was a batch.

    Parameters
    ----------
    results : list[R]
        The computed results, one per batch element.
    was_single : bool
        The was_single flag returned by ensure_batch().

    Returns
    -------
    R or list[R]
        results[0] if was_single, else results.
    """
    if was_single:
        return results[0]
    return results


def validate_batch_pair(target: list, data: list) -> None:
    """
    Validate that target and data have matching batch sizes.

    Parameters
    ----------
    target : list
        Batch of target items.
    data : list
        Batch of predicted items.

    Raises
    ------
    ValueError
        If the batch sizes differ.
    """
    if len(target) != len(data):
        raise ValueError(
            f"Batch size mismatch: target has {len(target)} item(s), "
            f"data has {len(data)} item(s)."
        )
