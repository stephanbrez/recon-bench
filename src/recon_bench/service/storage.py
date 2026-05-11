"""Filesystem storage helpers for service uploads and artifacts."""

import asyncio
import logging
import pathlib
import uuid

import fastapi

import recon_bench.service.config as service_config
import recon_bench.service.schemas as service_schemas


ARTIFACTS_DIR_NAME: str = "artifacts"
ARTIFACT_MEDIA_TYPE: str = "image/png"
ARTIFACT_ROUTE_PREFIX: str = "/v1/artifacts"
UPLOADS_DIR_NAME: str = "uploads"
UPLOAD_CHUNK_BYTES: int = 1024 * 1024

LOGGER: logging.Logger = logging.getLogger(__name__)


class StorageError(ValueError):
    """Raised when user-provided upload data fails validation."""


def ensure_service_dirs(config: service_config.ServiceConfig) -> None:
    """Create service runtime directories.

    Parameters
    ----------
    config
        Runtime settings that define storage and database locations.
    """
    config.storage_root.mkdir(parents=True, exist_ok=True)
    (config.storage_root / UPLOADS_DIR_NAME).mkdir(
        parents=True,
        exist_ok=True,
    )
    (config.storage_root / ARTIFACTS_DIR_NAME).mkdir(
        parents=True,
        exist_ok=True,
    )
    config.db_path.parent.mkdir(parents=True, exist_ok=True)


def validate_file_count(
    files: list[fastapi.UploadFile],
    config: service_config.ServiceConfig,
) -> None:
    """Validate uploaded file count.

    Parameters
    ----------
    files
        Files submitted for one multipart field.
    config
        Runtime settings containing the maximum file count.

    Raises
    ------
    StorageError
        If the request exceeds the configured file count.
    """
    if len(files) <= config.max_files:
        return
    raise StorageError(f"too many files: {len(files)} > {config.max_files}")


def validate_suffix(
    filename: str | None,
    allowed_suffixes: frozenset[str],
) -> str:
    """Validate an uploaded file suffix without trusting its path.

    Parameters
    ----------
    filename
        Client-provided filename. Only the suffix is read.
    allowed_suffixes
        Supported suffix set from the core library.

    Returns
    -------
    str
        Lowercase suffix to use for the generated server filename.

    Raises
    ------
    StorageError
        If the suffix is missing or unsupported.
    """
    suffix = pathlib.Path(filename or "").suffix.lower()
    if suffix in allowed_suffixes:
        return suffix
    raise StorageError(f"unsupported file suffix: {suffix or '<none>'}")


async def close_uploads(files: list[fastapi.UploadFile]) -> None:
    """Close upload handles, ignoring close failures.

    Parameters
    ----------
    files
        Upload handles that should not remain open after validation failure.
    """
    for file in files:
        try:
            await file.close()
        except Exception:
            LOGGER.exception("failed to close upload")


async def save_upload(
    file: fastapi.UploadFile,
    *,
    job_id: str,
    config: service_config.ServiceConfig,
    allowed_suffixes: frozenset[str],
) -> pathlib.Path:
    """Persist an uploaded file using a server-generated UUID name.

    Parameters
    ----------
    file
        Multipart upload provided by the client.
    job_id
        UUID string for the owning job.
    config
        Runtime settings containing storage and limit values.
    allowed_suffixes
        Supported suffix set for this upload field.

    Returns
    -------
    pathlib.Path
        Server-side path to the saved upload.

    Raises
    ------
    StorageError
        If suffix validation fails or upload size exceeds the configured
        limit.
    """
    try:
        suffix = validate_suffix(file.filename, allowed_suffixes)
        upload_dir = config.storage_root / UPLOADS_DIR_NAME / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{uuid.uuid4()}{suffix}"
        written = 0

        with path.open("wb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > config.max_upload_bytes:
                    path.unlink(missing_ok=True)
                    LOGGER.warning("upload too large", extra={"job_id": job_id})
                    raise StorageError("upload too large")
                await asyncio.to_thread(output.write, chunk)
    finally:
        await file.close()

    return path


def artifact_path(
    config: service_config.ServiceConfig,
    job_id: str,
    artifact_id: str,
) -> pathlib.Path:
    """Build the server-side artifact path.

    Parameters
    ----------
    config
        Runtime settings containing the storage root.
    job_id
        UUID string for the owning job.
    artifact_id
        UUID string for the generated artifact.

    Returns
    -------
    pathlib.Path
        PNG artifact path under the service artifact directory.
    """
    return (
        config.storage_root
        / ARTIFACTS_DIR_NAME
        / job_id
        / f"{artifact_id}.png"
    )


def artifact_out(
    *,
    artifact_id: str,
    role: str,
    index: int,
) -> service_schemas.ArtifactOut:
    """Create a response model for a saved artifact.

    Parameters
    ----------
    artifact_id
        UUID string for the artifact.
    role
        Render role from the evaluator, either ``target`` or ``prediction``.
    index
        View index within the rendered image batch.

    Returns
    -------
    service_schemas.ArtifactOut
        JSON-safe artifact response model.
    """
    if role not in service_schemas.ARTIFACT_ROLES:
        raise ValueError(f"unsupported artifact role: {role}")
    return service_schemas.ArtifactOut(
        artifact_id=artifact_id,
        role=role,
        index=index,
        media_type=ARTIFACT_MEDIA_TYPE,
        url=f"{ARTIFACT_ROUTE_PREFIX}/{artifact_id}",
    )
