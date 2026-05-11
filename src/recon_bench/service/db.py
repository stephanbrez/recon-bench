"""SQLite persistence helpers for the optional service."""

import contextlib
import datetime
import json
import pathlib
import typing

import aiosqlite

import recon_bench.service.schemas as service_schemas


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JobRow = aiosqlite.Row
ArtifactRecord = tuple[service_schemas.ArtifactOut, pathlib.Path]


def utcnow() -> datetime.datetime:
    """Return an aware UTC timestamp.

    Returns
    -------
    datetime.datetime
        Current UTC time with timezone information.
    """
    return datetime.datetime.now(datetime.UTC)


def _to_json(data: JsonValue) -> str:
    return json.dumps(data, separators=(",", ":"))


class Database:
    """Async SQLite persistence layer for service jobs.

    Parameters
    ----------
    path
        Filesystem path to the service SQLite database.
    """

    path: pathlib.Path

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path

    @contextlib.asynccontextmanager
    async def _connect(self) -> typing.AsyncIterator[aiosqlite.Connection]:
        db = await aiosqlite.connect(self.path)
        try:
            await db.execute("PRAGMA foreign_keys = ON")
            yield db
        finally:
            await db.close()

    async def init(self) -> None:
        """Initialize the SQLite schema if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as db:
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
        mode: service_schemas.EvalMode,
        request: dict[str, JsonValue],
        created_at: datetime.datetime,
    ) -> None:
        """Insert a pending job row.

        Parameters
        ----------
        job_id
            UUID string for the job.
        mode
            Evaluation mode requested by the client.
        request
            JSON-safe request options for diagnostics.
        created_at
            Job creation timestamp.
        """
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO jobs
                    (job_id, mode, status, request_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    mode.value,
                    service_schemas.JobStatus.PENDING.value,
                    _to_json(request),
                    created_at.isoformat(),
                ),
            )
            await db.commit()

    async def set_status(
        self,
        job_id: str,
        status: service_schemas.JobStatus,
        *,
        completed_at: datetime.datetime | None = None,
        error: dict[str, JsonValue] | None = None,
    ) -> None:
        """Update a job lifecycle status.

        Parameters
        ----------
        job_id
            UUID string for the job.
        status
            New lifecycle state.
        completed_at
            Optional completion timestamp.
        error
            Optional JSON-safe error details.
        """
        async with self._connect() as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = ?,
                    completed_at = COALESCE(?, completed_at),
                    error_json = ?
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
        metrics: list[service_schemas.MetricValue] | None,
    ) -> None:
        """Insert metric values for a completed job."""
        if not metrics:
            return

        rows = [
            (job_id, family, metric.name, index, value)
            for metric in metrics
            for index, value in enumerate(metric.values)
        ]
        async with self._connect() as db:
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
        artifacts: list[ArtifactRecord],
    ) -> None:
        """Insert generated artifact records for a completed job."""
        if not artifacts:
            return

        async with self._connect() as db:
            await db.executemany(
                """
                INSERT INTO artifacts
                    (artifact_id, job_id, role, item_index, media_type, path)
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

    async def get_job(self, job_id: str) -> JobRow | None:
        """Fetch a job row by ID.

        Parameters
        ----------
        job_id
            UUID string for the job.

        Returns
        -------
        JobRow or None
            SQLite row when found.
        """
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            )
            return await cursor.fetchone()

    async def get_artifact_path(self, artifact_id: str) -> pathlib.Path | None:
        """Fetch a generated artifact path by artifact ID."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT path FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = await cursor.fetchone()
        return pathlib.Path(row[0]) if row else None
