"""Service configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    storage_root: Path = Path("runs/service")
    db_path: Path = Path("runs/service/service.sqlite3")
    gpu_slots: int = 1
    max_upload_bytes: int = 512 * 1024 * 1024
    max_files: int = 128


def get_config() -> ServiceConfig:
    return ServiceConfig()
