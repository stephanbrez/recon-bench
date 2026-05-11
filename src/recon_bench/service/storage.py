"""Filesystem storage helpers for service uploads and artifacts."""

from __future__ import annotations

import pathlib
import uuid

from fastapi import UploadFile

from recon_bench import _types

from .config import ServiceConfig
from .schemas import ArtifactOut


class StorageError(ValueError):
    pass


def ensure_service_dirs(config: ServiceConfig) -> None:
    config.storage_root.mkdir(parents=True, exist_ok=True)
    (config.storage_root / "uploads").mkdir(parents=True, exist_ok=True)
    (config.storage_root / "artifacts").mkdir(parents=True, exist_ok=True)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)


def validate_file_count(files: list[UploadFile], config: ServiceConfig) -> None:
    if len(files) > config.max_files:
        raise StorageError(f"too many files: {len(files)} > {config.max_files}")


def validate_suffix(filename: str | None, allowed_suffixes: frozenset[str]) -> str:
    suffix = pathlib.Path(filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        raise StorageError(f"unsupported file suffix: {suffix or '<none>'}")
    return suffix


def validate_image_suffix(filename: str | None) -> str:
    return validate_suffix(filename, _types.IMAGE_SUFFIXES)


def validate_geometry_suffix(filename: str | None) -> str:
    return validate_suffix(filename, _types.MESH_SUFFIXES)


async def save_upload(
    file: UploadFile,
    *,
    job_id: str,
    config: ServiceConfig,
    allowed_suffixes: frozenset[str],
) -> pathlib.Path:
    suffix = validate_suffix(file.filename, allowed_suffixes)
    upload_dir = config.storage_root / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4()}{suffix}"

    written = 0
    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > config.max_upload_bytes:
                path.unlink(missing_ok=True)
                raise StorageError("upload too large")
            output.write(chunk)
    await file.close()
    return path


def artifact_path(config: ServiceConfig, job_id: str, artifact_id: str) -> pathlib.Path:
    return config.storage_root / "artifacts" / job_id / f"{artifact_id}.png"


def artifact_out(
    *,
    artifact_id: str,
    role: str,
    index: int,
) -> ArtifactOut:
    if role not in {"target", "prediction"}:
        raise ValueError(f"unsupported artifact role: {role}")
    return ArtifactOut(
        artifact_id=artifact_id,
        role=role,
        index=index,
        media_type="image/png",
        url=f"/v1/artifacts/{artifact_id}",
    )
