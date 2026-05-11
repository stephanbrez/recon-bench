"""SQLite persistence helpers for the optional service."""

from __future__ import annotations

import datetime
import json
import pathlib
import typing

import aiosqlite

from .schemas import ArtifactOut, EvalMode, JobStatus, MetricValue


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _to_json(data: object) -> str:
    return json.dumps(data, default=str, separators=(",", ":"))


class Database:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    job_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    name TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    value REAL NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                """
            )
            await db.commit()

    async def create_job(
        self,
        *,
        job_id: str,
        mode: EvalMode,
        request: dict[str, typing.Any],
        created_at: datetime.datetime,
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO jobs (job_id, mode, status, request_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    mode.value,
                    JobStatus.PENDING.value,
                    _to_json(request),
                    created_at.isoformat(),
                ),
            )
            await db.commit()

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        completed_at: datetime.datetime | None = None,
        error: dict[str, typing.Any] | None = None,
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = COALESCE(?, completed_at), error_json = ?
                WHERE job_id = ?
                """,
                (
                    status.value,
                    completed_at.isoformat() if completed_at else None,
                    _to_json(error) if error is not None else None,
                    job_id,
                ),
            )
            await db.commit()

    async def insert_metrics(
        self,
        *,
        job_id: str,
        family: str,
        metrics: list[MetricValue] | None,
    ) -> None:
        if not metrics:
            return
        rows = [
            (job_id, family, metric.name, index, value)
            for metric in metrics
            for index, value in enumerate(metric.values)
        ]
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                """
                INSERT INTO metrics (job_id, family, name, item_index, value)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            await db.commit()

    async def insert_artifacts(
        self,
        *,
        job_id: str,
        artifacts: list[tuple[ArtifactOut, pathlib.Path]],
    ) -> None:
        if not artifacts:
            return
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                """
                INSERT INTO artifacts (artifact_id, job_id, role, item_index, media_type, path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        artifact.artifact_id,
                        job_id,
                        artifact.role,
                        artifact.index,
                        artifact.media_type,
                        str(path),
                    )
                    for artifact, path in artifacts
                ],
            )
            await db.commit()

    async def get_job(self, job_id: str) -> aiosqlite.Row | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            return await cursor.fetchone()

    async def get_artifact_path(self, artifact_id: str) -> pathlib.Path | None:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT path FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = await cursor.fetchone()
        return pathlib.Path(row[0]) if row else None
