"""HTTP routes for the optional service."""

from __future__ import annotations

import datetime
import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from recon_bench import _types

from . import db as db_mod
from .schemas import (
    ErrorDetail,
    ErrorResponse,
    EvalMode,
    EvalResponse,
    ImageVsImageOptions,
    ImageVsMeshOptions,
    JobStatus,
    MeshVsMeshOptions,
)
from .serialization import metrics_to_out, profile_to_out, run_evaluate, save_render_artifacts
from .storage import StorageError, save_upload, validate_file_count


router = APIRouter()


def _state(request: Request):
    return request.app.state.config, request.app.state.db, request.app.state.gpu_semaphore


def _parse_options(raw: str, model_type):
    try:
        return model_type.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def _error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    job_id: str | None = None,
    details: dict[str, object] | None = None,
) -> HTTPException:
    error = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details or {},
            job_id=job_id,
        )
    )
    return HTTPException(status_code=status_code, detail=error.model_dump())


async def _save_uploads(files: list[UploadFile], *, job_id: str, config, suffixes):
    validate_file_count(files, config)
    return [
        await save_upload(file, job_id=job_id, config=config, allowed_suffixes=suffixes)
        for file in files
    ]


async def _finalize_success(
    *,
    db: db_mod.Database,
    job_id: str,
    mode: EvalMode,
    result,
    created_at: datetime.datetime,
    completed_at: datetime.datetime,
    artifacts,
) -> EvalResponse:
    metrics = metrics_to_out(result)
    await db.insert_metrics(job_id=job_id, family="image", metrics=metrics.image)
    await db.insert_metrics(job_id=job_id, family="geometry", metrics=metrics.geometry)
    await db.insert_artifacts(job_id=job_id, artifacts=artifacts)
    await db.set_status(job_id, JobStatus.COMPLETED, completed_at=completed_at)
    return EvalResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        mode=mode,
        metrics=metrics,
        profile=profile_to_out(result.profile),
        artifacts=[artifact for artifact, _ in artifacts],
        created_at=created_at,
        completed_at=completed_at,
    )


async def _mark_failed(db: db_mod.Database, job_id: str, exc: Exception, code: str) -> None:
    await db.set_status(
        job_id,
        JobStatus.FAILED,
        completed_at=db_mod.utcnow(),
        error={"code": code, "message": str(exc)},
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/evals/image-vs-image", response_model=EvalResponse)
async def image_vs_image(
    request: Request,
    target_files: list[UploadFile] = File(...),
    prediction_files: list[UploadFile] = File(...),
    options: str = Form("{}"),
) -> EvalResponse:
    config, db, semaphore = _state(request)
    parsed = _parse_options(options, ImageVsImageOptions)
    if len(target_files) != len(prediction_files) or not target_files:
        raise _error_response(400, code="invalid_files", message="target and prediction image counts must match")

    job_id = str(uuid.uuid4())
    created_at = db_mod.utcnow()
    await db.create_job(job_id=job_id, mode=EvalMode.IMAGE_VS_IMAGE, request=parsed.model_dump(mode="json"), created_at=created_at)
    try:
        targets = await _save_uploads(target_files, job_id=job_id, config=config, suffixes=_types.IMAGE_SUFFIXES)
        predictions = await _save_uploads(prediction_files, job_id=job_id, config=config, suffixes=_types.IMAGE_SUFFIXES)
        await db.set_status(job_id, JobStatus.RUNNING)
        result = await run_evaluate(
            semaphore,
            target=targets,
            prediction=predictions,
            mode="image_vs_image",
            image_metrics=parsed.metrics,
            profile=parsed.profile,
            shard_size=parsed.shard_size,
            max_size=parsed.max_size,
            background_color=parsed.normalized_background_color(),
        )
        return await _finalize_success(
            db=db,
            job_id=job_id,
            mode=EvalMode.IMAGE_VS_IMAGE,
            result=result,
            created_at=created_at,
            completed_at=db_mod.utcnow(),
            artifacts=[],
        )
    except StorageError as exc:
        await _mark_failed(db, job_id, exc, "invalid_upload")
        raise _error_response(400, code="invalid_upload", message=str(exc), job_id=job_id) from exc
    except (ValueError, TypeError) as exc:
        await _mark_failed(db, job_id, exc, "evaluation_failed")
        raise _error_response(400, code="evaluation_failed", message=str(exc), job_id=job_id) from exc


@router.post("/v1/evals/image-vs-mesh", response_model=EvalResponse)
async def image_vs_mesh(
    request: Request,
    target_files: list[UploadFile] = File(...),
    prediction_file: UploadFile = File(...),
    options: str = Form(...),
) -> EvalResponse:
    config, db, semaphore = _state(request)
    parsed = _parse_options(options, ImageVsMeshOptions)
    cameras = parsed.camera_input()
    expected_targets = len(cameras) if isinstance(cameras, list) else 1
    if len(target_files) != expected_targets:
        raise _error_response(400, code="invalid_files", message="target image count must match camera count")

    job_id = str(uuid.uuid4())
    created_at = db_mod.utcnow()
    await db.create_job(job_id=job_id, mode=EvalMode.IMAGE_VS_MESH, request=parsed.model_dump(mode="json"), created_at=created_at)
    try:
        targets = await _save_uploads(target_files, job_id=job_id, config=config, suffixes=_types.IMAGE_SUFFIXES)
        prediction = (await _save_uploads([prediction_file], job_id=job_id, config=config, suffixes=_types.MESH_SUFFIXES))[0]
        await db.set_status(job_id, JobStatus.RUNNING)
        result = await run_evaluate(
            semaphore,
            target=targets if len(targets) > 1 else targets[0],
            prediction=prediction,
            mode="image_vs_mesh",
            camera=cameras,
            image_metrics=parsed.metrics,
            profile=parsed.profile,
            shard_size=parsed.shard_size,
            max_size=parsed.max_size,
            background_color=parsed.normalized_background_color(),
        )
        artifacts = []
        if parsed.save_renders:
            await db.set_status(job_id, JobStatus.SAVING_ARTIFACTS)
            artifacts = save_render_artifacts(result, job_id=job_id, config=config)
        return await _finalize_success(db=db, job_id=job_id, mode=EvalMode.IMAGE_VS_MESH, result=result, created_at=created_at, completed_at=db_mod.utcnow(), artifacts=artifacts)
    except StorageError as exc:
        await _mark_failed(db, job_id, exc, "invalid_upload")
        raise _error_response(400, code="invalid_upload", message=str(exc), job_id=job_id) from exc
    except (ValueError, TypeError) as exc:
        await _mark_failed(db, job_id, exc, "evaluation_failed")
        raise _error_response(400, code="evaluation_failed", message=str(exc), job_id=job_id) from exc


@router.post("/v1/evals/mesh-vs-mesh", response_model=EvalResponse)
async def mesh_vs_mesh(
    request: Request,
    target_file: UploadFile = File(...),
    prediction_file: UploadFile = File(...),
    options: str = Form("{}"),
) -> EvalResponse:
    config, db, semaphore = _state(request)
    parsed = _parse_options(options, MeshVsMeshOptions)
    job_id = str(uuid.uuid4())
    created_at = db_mod.utcnow()
    await db.create_job(job_id=job_id, mode=EvalMode.MESH_VS_MESH, request=parsed.model_dump(mode="json"), created_at=created_at)
    try:
        target = (await _save_uploads([target_file], job_id=job_id, config=config, suffixes=_types.MESH_SUFFIXES))[0]
        prediction = (await _save_uploads([prediction_file], job_id=job_id, config=config, suffixes=_types.MESH_SUFFIXES))[0]
        await db.set_status(job_id, JobStatus.RUNNING)
        result = await run_evaluate(
            semaphore,
            target=target,
            prediction=prediction,
            mode="mesh_vs_mesh",
            camera=parsed.camera_input(),
            image_metrics=parsed.metrics,
            image_eval=parsed.image_eval,
            geometry_type=parsed.geometry_type,
            num_points=parsed.num_points,
            profile=parsed.profile,
            shard_size=parsed.shard_size,
            max_size=parsed.max_size,
            background_color=parsed.normalized_background_color(),
        )
        artifacts = []
        if parsed.save_renders:
            await db.set_status(job_id, JobStatus.SAVING_ARTIFACTS)
            artifacts = save_render_artifacts(result, job_id=job_id, config=config)
        return await _finalize_success(db=db, job_id=job_id, mode=EvalMode.MESH_VS_MESH, result=result, created_at=created_at, completed_at=db_mod.utcnow(), artifacts=artifacts)
    except StorageError as exc:
        await _mark_failed(db, job_id, exc, "invalid_upload")
        raise _error_response(400, code="invalid_upload", message=str(exc), job_id=job_id) from exc
    except (ValueError, TypeError) as exc:
        await _mark_failed(db, job_id, exc, "evaluation_failed")
        raise _error_response(400, code="evaluation_failed", message=str(exc), job_id=job_id) from exc


@router.get("/v1/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    _, db, _ = _state(request)
    row = await db.get_job(job_id)
    if row is None:
        raise _error_response(404, code="not_found", message="job not found")
    return dict(row)


@router.get("/v1/jobs/{job_id}/result")
async def get_job_result(request: Request, job_id: str):
    _, db, _ = _state(request)
    row = await db.get_job(job_id)
    if row is None:
        raise _error_response(404, code="not_found", message="job not found")
    if row["status"] != JobStatus.COMPLETED.value:
        raise _error_response(400, code="job_not_completed", message="job result is not available yet", job_id=job_id)
    return dict(row)


@router.get("/v1/artifacts/{artifact_id}")
async def get_artifact(request: Request, artifact_id: str) -> FileResponse:
    _, db, _ = _state(request)
    path = await db.get_artifact_path(artifact_id)
    if path is None or not path.exists():
        raise _error_response(404, code="not_found", message="artifact not found")
    return FileResponse(path, media_type="image/png")
