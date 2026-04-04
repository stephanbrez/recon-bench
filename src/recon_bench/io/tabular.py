import csv
import json
import pathlib
from typing import Any, Sequence


def write_to_csv(
    path: str | pathlib.Path | None,
    data: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    """
    Write a sequence of dictionaries to a CSV file.

    It explicitly maps column names to values and gracefully handles optional
    fields.

    Parameters
    ----------
    path : str, pathlib.Path, or None
        Destination file path. If None or empty string, nothing is written.
    data : Sequence[dict[str, Any]]
        Data rows to write.
    fieldnames : Sequence[str], optional
        Explicit list of columns to write. If not provided, it will be
        inferred by collecting all unique keys from the dictionaries.
    """
    if not path:
        raise ValueError("Must specify an output file")

    if not data and not fieldnames:
        # Nothing to write and no headers to infer
        raise ValueError("Missing data to write")

    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Infer fieldnames by preserving insertion order of all seen keys
    if fieldnames is None:
        keys_dict = {}
        for row in data:
            for k in row.keys():
                keys_dict[k] = None
        fieldnames = list(keys_dict.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
        writer.writerows(data)


def write_to_json(
    path: str | pathlib.Path | None,
    data: Sequence[dict[str, Any]],
) -> None:
    """
    Write a sequence of dictionaries to a JSON file.

    Parameters
    ----------
    path : str, pathlib.Path, or None
        Destination file path. If None or empty string, nothing is written.
    data : Sequence[dict[str, Any]]
        Data payload to serialize as JSON.
    """
    if not path:
        raise ValueError("Must specify an output file")

    if not data:
        raise ValueError("Missing data to write")

    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
