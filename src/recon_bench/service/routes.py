"""HTTP routes for the optional service."""

import asyncio
import dataclasses
import datetime
import json
import logging
import pathlib
import typing
import uuid

import fastapi
import fastapi.responses
import pydantic

import recon_bench
import recon_bench._types as recon_types
import recon_bench.service.config as service_config
import recon_bench.service.db as service_db
import recon_bench.service.schemas as service_schemas
import recon_bench.service.serialization as service_serialization
import recon_bench.service.storage as service_storage


EVALS_PREFIX: str = "/v1/evals"
HEALTH_PATH: str = "/health"
JOBS_PREFIX: str = "/v1/jobs"
INTERNAL_ERROR_MESSAGE: str = "Internal server error."

LOGGER: logging.Logger = logging.getLogger(__name__)
router: fastapi.APIRouter = fastapi.APIRouter()

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
OptionsModel = typing.TypeVar("OptionsModel", bound=pydantic.BaseModel)
Options = (
    service_schemas.ImageVsImageOptions
    | service_schemas.ImageVsMeshOptions
    | service_schemas.MeshVsMeshOptions
)


@dataclasses.dataclass(frozen=True)
class ServiceState:
    """Typed view of FastAPI application state.

    Parameters
    ----------
    config
        Runtime service settings.
    db
        SQLite persistence layer.
    semaphore
        GPU concurrency guard.
    """

    config: service_config.ServiceConfig
    db: service_db.Database
    semaphore: asyncio.Semaphore


@dataclasses.dataclass(frozen=True)
class PreparedEvaluation:
    """Mode-specific data required to run an evaluation.

    Parameters
    ----------
    mode
        Service mode used for persistence and response metadata.
    evaluate_kwargs
        Keyword arguments forwarded to ``recon_bench.evaluate``.
    save_renders
        Whether render artifacts should be persisted after evaluation.
    """

    mode: service_schemas.EvalMode
    evaluate_kwargs: dict[str, object]
    save_renders: bool


def _state(request: fastapi.Request) -> ServiceState:
    config = typing.cast(
        service_config.ServiceConfig,
        request.app.state.config,
    )
    semaphore = typing.cast(
        asyncio.Semaphore,
        request.app.state.gpu_semaphore,
    )
    return ServiceState(
        config=config,
        db=typing.cast(service_db.Database, request.app.state.db),
        semaphore=semaphore,
    )


def _parse_options(raw: str, model_type: type[OptionsModel]) -> OptionsModel:
    try:
        return model_type.model_validate_json(raw)
    except pydantic.ValidationError as exc:
        raise fastapi.HTTPException(
            status_code=422,
            detail=json.loads(exc.json()),
        ) from exc


def _error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    job_id: str | None = None,
    details: dict[str, object] | None = None,
) -> fastapi.HTTPException:
    error = service_schemas.ErrorResponse(
        error=service_schemas.ErrorDetail(
            code=code,
            message=message,
            details=details or {},
            job_id=job_id,
        )
    )
    return fastapi.HTTPException(
        status_code=status_code,
        detail=error.model_dump(),
    )


def _request_payload(options: Options) -> dict[str, JsonValue]:
    return typing.cast(dict[str, JsonValue], options.model_dump(mode="json"))


async def _save_uploads(
    files: list[fastapi.UploadFile],
    *,
    job_id: str,
    config: service_config.ServiceConfig,
    suffixes: frozenset[str],
) -> list[pathlib.Path]:
    try:
        service_storage.validate_file_count(files, config)
        return [
            await service_storage.save_upload(
                file,
                job_id=job_id,
                config=config,
                allowed_suffixes=suffixes,
            )
            for file in files
        ]
    except BaseException:
        await service_storage.close_uploads(files)
        raise


async def _mark_failed(
    db: service_db.Database,
    job_id: str,
    exc: BaseException,
    code: str,
) -> None:
    await db.set_status(
        job_id,
        service_schemas.JobStatus.FAILED,
        completed_at=service_db.utcnow(),
        error={
            "code": code,
            "message": str(exc),
        },
    )


async def _save_artifacts(
    *,
    state: ServiceState,
    job_id: str,
    result: recon_bench.EvalResult,
) -> list[service_serialization.ArtifactRecord]:
    await state.db.set_status(
        job_id,
        service_schemas.JobStatus.SAVING_ARTIFACTS,
    )
    return await asyncio.to_thread(
        service_serialization.save_render_artifacts,
        result,
        job_id=job_id,
        config=state.config,
    )


async def _finalize_success(
    *,
    state: ServiceState,
    job_id: str,
    mode: service_schemas.EvalMode,
    result: recon_bench.EvalResult,
    created_at: datetime.datetime,
    artifacts: list[service_serialization.ArtifactRecord],
) -> service_schemas.EvalResponse:
    completed_at = service_db.utcnow()
    metrics = service_serialization.metrics_to_out(result)
    await state.db.insert_metrics(
        job_id=job_id,
        family="image",
        metrics=metrics.image,
    )
    await state.db.insert_metrics(
        job_id=job_id,
        family="geometry",
        metrics=metrics.geometry,
    )
    await state.db.insert_artifacts(job_id=job_id, artifacts=artifacts)
    await state.db.set_status(
        job_id,
        service_schemas.JobStatus.COMPLETED,
        completed_at=completed_at,
    )

    return service_schemas.EvalResponse(
        job_id=job_id,
        status=service_schemas.JobStatus.COMPLETED,
        mode=mode,
        metrics=metrics,
        profile=service_serialization.profile_to_out(result.profile),
        artifacts=[artifact for artifact, _ in artifacts],
        created_at=created_at,
        completed_at=completed_at,
    )


async def _execute_job(
    *,
    state: ServiceState,
    job_id: str,
    options: Options,
    prepared: PreparedEvaluation,
) -> service_schemas.EvalResponse:
    """Run a full job lifecycle with consistent error handling.

    Parameters
    ----------
    state
        Runtime app state.
    job_id
        UUID string shared by upload storage and persistence.
    options
        Validated request options.
    prepared
        Mode-specific evaluator arguments.

    Returns
    -------
    service_schemas.EvalResponse
        Final response for a completed evaluation.
    """
    created_at = service_db.utcnow()
    await state.db.create_job(
        job_id=job_id,
        mode=prepared.mode,
        request=_request_payload(options),
        created_at=created_at,
    )

    try:
        LOGGER.info("job running", extra={"job_id": job_id})
        await state.db.set_status(job_id, service_schemas.JobStatus.RUNNING)
        result = await service_serialization.run_evaluate(
            state.semaphore,
            prepared.evaluate_kwargs,
        )
    except (ValueError, TypeError) as exc:
        await _mark_failed(state.db, job_id, exc, "evaluation_failed")
        LOGGER.info("job failed", extra={"job_id": job_id})
        raise _error_response(
            400,
            code="evaluation_failed",
            message=str(exc),
            job_id=job_id,
        ) from exc
    except asyncio.CancelledError as exc:
        await _mark_failed(state.db, job_id, exc, "request_cancelled")
        LOGGER.warning("job cancelled", extra={"job_id": job_id})
        raise
    except Exception as exc:
        await _mark_failed(state.db, job_id, exc, "internal_error")
        LOGGER.exception("job failed unexpectedly", extra={"job_id": job_id})
        raise _error_response(
            500,
            code="internal_error",
            message=INTERNAL_ERROR_MESSAGE,
            job_id=job_id,
        ) from exc

    try:
        artifacts = []
        if prepared.save_renders:
            artifacts = await _save_artifacts(
                state=state,
                job_id=job_id,
                result=result,
            )
        response = await _finalize_success(
            state=state,
            job_id=job_id,
            mode=prepared.mode,
            result=result,
            created_at=created_at,
            artifacts=artifacts,
        )
        LOGGER.info("job completed", extra={"job_id": job_id})
        return response
    except Exception as exc:
        await _mark_failed(state.db, job_id, exc, "artifact_or_persist_failed")
        LOGGER.exception("job finalization failed", extra={"job_id": job_id})
        raise _error_response(
            500,
            code="artifact_or_persist_failed",
            message=INTERNAL_ERROR_MESSAGE,
            job_id=job_id,
        ) from exc


@router.get(HEALTH_PATH)
async def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@router.post(
    f"{EVALS_PREFIX}/image-vs-image",
    response_model=service_schemas.EvalResponse,
)
async def image_vs_image(
    request: fastapi.Request,
    target_files: list[fastapi.UploadFile] = fastapi.File(...),
    prediction_files: list[fastapi.UploadFile] = fastapi.File(...),
    options: str = fastapi.Form("{}"),
) -> service_schemas.EvalResponse:
    """Evaluate uploaded prediction images against target images."""
    state = _state(request)
    parsed = _parse_options(options, service_schemas.ImageVsImageOptions)

    if len(target_files) != len(prediction_files) or not target_files:
        await service_storage.close_uploads(target_files + prediction_files)
        raise _error_response(
            400,
            code="invalid_files",
            message="target and prediction image counts must match",
        )

    job_id = str(uuid.uuid4())
    try:
        targets = await _save_uploads(
            target_files,
            job_id=job_id,
            config=state.config,
            suffixes=recon_types.IMAGE_SUFFIXES,
        )
        predictions = await _save_uploads(
            prediction_files,
            job_id=job_id,
            config=state.config,
            suffixes=recon_types.IMAGE_SUFFIXES,
        )
    except service_storage.StorageError as exc:
        raise _error_response(
            400,
            code="invalid_upload",
            message=str(exc),
        ) from exc

    return await _execute_job(
        state=state,
        job_id=job_id,
        options=parsed,
        prepared=PreparedEvaluation(
            mode=service_schemas.EvalMode.IMAGE_VS_IMAGE,
            evaluate_kwargs={
                "target": targets,
                "prediction": predictions,
                "mode": "image_vs_image",
                "image_metrics": parsed.metrics,
                "profile": parsed.profile,
                "shard_size": parsed.shard_size,
                "max_size": parsed.max_size,
                "background_color": parsed.normalized_background_color(),
            },
            save_renders=False,
        ),
    )


@router.post(
    f"{EVALS_PREFIX}/image-vs-mesh",
    response_model=service_schemas.EvalResponse,
)
async def image_vs_mesh(
    request: fastapi.Request,
    target_files: list[fastapi.UploadFile] = fastapi.File(...),
    prediction_file: fastapi.UploadFile = fastapi.File(...),
    options: str = fastapi.Form(...),
) -> service_schemas.EvalResponse:
    """Evaluate uploaded target images against an uploaded mesh."""
    state = _state(request)
    parsed = _parse_options(options, service_schemas.ImageVsMeshOptions)
    cameras = parsed.camera_input()
    expected_targets = len(cameras) if isinstance(cameras, list) else 1

    if len(target_files) != expected_targets:
        await service_storage.close_uploads(target_files + [prediction_file])
        raise _error_response(
            400,
            code="invalid_files",
            message="target image count must match camera count",
        )

    job_id = str(uuid.uuid4())
    try:
        targets = await _save_uploads(
            target_files,
            job_id=job_id,
            config=state.config,
            suffixes=recon_types.IMAGE_SUFFIXES,
        )
        predictions = await _save_uploads(
            [prediction_file],
            job_id=job_id,
            config=state.config,
            suffixes=recon_types.MESH_SUFFIXES,
        )
    except service_storage.StorageError as exc:
        raise _error_response(
            400,
            code="invalid_upload",
            message=str(exc),
        ) from exc

    return await _execute_job(
        state=state,
        job_id=job_id,
        options=parsed,
        prepared=PreparedEvaluation(
            mode=service_schemas.EvalMode.IMAGE_VS_MESH,
            evaluate_kwargs={
                "target": targets if len(targets) > 1 else targets[0],
                "prediction": predictions[0],
                "mode": "image_vs_mesh",
                "camera": cameras,
                "image_metrics": parsed.metrics,
                "profile": parsed.profile,
                "shard_size": parsed.shard_size,
                "max_size": parsed.max_size,
                "background_color": parsed.normalized_background_color(),
            },
            save_renders=parsed.save_renders,
        ),
    )


@router.post(
    f"{EVALS_PREFIX}/mesh-vs-mesh",
    response_model=service_schemas.EvalResponse,
)
async def mesh_vs_mesh(
    request: fastapi.Request,
    target_file: fastapi.UploadFile = fastapi.File(...),
    prediction_file: fastapi.UploadFile = fastapi.File(...),
    options: str = fastapi.Form("{}"),
) -> service_schemas.EvalResponse:
    """Evaluate uploaded geometry files against each other."""
    state = _state(request)
    parsed = _parse_options(options, service_schemas.MeshVsMeshOptions)
    job_id = str(uuid.uuid4())

    try:
        targets = await _save_uploads(
            [target_file],
            job_id=job_id,
            config=state.config,
            suffixes=recon_types.MESH_SUFFIXES,
        )
        predictions = await _save_uploads(
            [prediction_file],
            job_id=job_id,
            config=state.config,
            suffixes=recon_types.MESH_SUFFIXES,
        )
    except service_storage.StorageError as exc:
        raise _error_response(
            400,
            code="invalid_upload",
            message=str(exc),
        ) from exc

    return await _execute_job(
        state=state,
        job_id=job_id,
        options=parsed,
        prepared=PreparedEvaluation(
            mode=service_schemas.EvalMode.MESH_VS_MESH,
            evaluate_kwargs={
                "target": targets[0],
                "prediction": predictions[0],
                "mode": "mesh_vs_mesh",
                "camera": parsed.camera_input(),
                "image_metrics": parsed.metrics,
                "image_eval": parsed.image_eval,
                "geometry_type": parsed.geometry_type,
                "num_points": parsed.num_points,
                "profile": parsed.profile,
                "shard_size": parsed.shard_size,
                "max_size": parsed.max_size,
                "background_color": parsed.normalized_background_color(),
            },
            save_renders=parsed.save_renders,
        ),
    )


@router.get(f"{JOBS_PREFIX}/{{job_id}}")
async def get_job(request: fastapi.Request, job_id: str) -> dict[str, object]:
    """Return persisted job metadata."""
    row = await _state(request).db.get_job(job_id)
    if row is not None:
        return dict(row)
    raise _error_response(404, code="not_found", message="job not found")


@router.get(f"{JOBS_PREFIX}/{{job_id}}/result")
async def get_job_result(
    request: fastapi.Request,
    job_id: str,
) -> dict[str, object]:
    """Return persisted completed-job metadata."""
    row = await _state(request).db.get_job(job_id)
    if row is None:
        raise _error_response(404, code="not_found", message="job not found")
    if row["status"] == service_schemas.JobStatus.COMPLETED.value:
        return dict(row)
    raise _error_response(
        400,
        code="job_not_completed",
        message="job result is not available yet",
        job_id=job_id,
    )


@router.get(f"{service_storage.ARTIFACT_ROUTE_PREFIX}/{{artifact_id}}")
async def get_artifact(
    request: fastapi.Request,
    artifact_id: str,
) -> fastapi.responses.FileResponse:
    """Return a generated PNG artifact."""
    path = await _state(request).db.get_artifact_path(artifact_id)
    if path is not None and path.exists():
        return fastapi.responses.FileResponse(
            path,
            media_type=service_storage.ARTIFACT_MEDIA_TYPE,
        )
    raise _error_response(404, code="not_found", message="artifact not found")
