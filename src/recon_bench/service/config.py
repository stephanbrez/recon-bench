"""Service configuration."""

import dataclasses
import pathlib


DEFAULT_STORAGE_ROOT: pathlib.Path = pathlib.Path("runs/service")
DEFAULT_DB_PATH: pathlib.Path = pathlib.Path("runs/service/service.sqlite3")
DEFAULT_GPU_SLOTS: int = 1
DEFAULT_MAX_UPLOAD_BYTES: int = 512 * 1024 * 1024
DEFAULT_MAX_FILES: int = 128


@dataclasses.dataclass(frozen=True)
class ServiceConfig:
    """Runtime settings for the optional service.

    Parameters
    ----------
    storage_root
        Base directory for service uploads and generated artifacts.
    db_path
        SQLite database path used for service job history.
    gpu_slots
        Number of concurrent evaluations allowed to enter GPU-heavy work.
    max_upload_bytes
        Maximum accepted size for each uploaded file.
    max_files
        Maximum number of files accepted for a single upload field.
    """

    storage_root: pathlib.Path = DEFAULT_STORAGE_ROOT
    db_path: pathlib.Path = DEFAULT_DB_PATH
    gpu_slots: int = DEFAULT_GPU_SLOTS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_files: int = DEFAULT_MAX_FILES

    def __post_init__(self) -> None:
        """Validate numeric service settings.

        Raises
        ------
        ValueError
            If concurrency or upload limits cannot produce a working service.
        """
        if self.gpu_slots < 1:
            raise ValueError("gpu_slots must be at least 1")
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be at least 1")
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1")


def get_config() -> ServiceConfig:
    """Return service settings.

    Returns
    -------
    ServiceConfig
        Default service settings. Environment-based configuration is deferred
        intentionally and will be added separately.
    """
    return ServiceConfig()
