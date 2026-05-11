"""JSON-safe conversion helpers for evaluation results."""

import asyncio
import dataclasses
import pathlib
import typing
import uuid

import torch

import recon_bench
import recon_bench.io.image as io_image
import recon_bench.service.config as service_config
import recon_bench.service.schemas as service_schemas
import recon_bench.service.storage as service_storage


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ArtifactRecord = tuple[service_schemas.ArtifactOut, pathlib.Path]


def tensor_metrics_to_out(
    metrics: dict[str, torch.Tensor] | None,
) -> list[service_schemas.MetricValue] | None:
    """Convert metric tensors to JSON-safe lists.

    Parameters
    ----------
    metrics
        Metric mapping returned by ``recon_bench.evaluate``.

    Returns
    -------
    list[service_schemas.MetricValue] or None
        Per-metric values converted to Python floats.
    """
    if metrics is None:
        return None
    return [
        service_schemas.MetricValue(
            name=name,
            values=[
                float(value)
                for value in tensor.detach().cpu().flatten().tolist()
            ],
        )
        for name, tensor in metrics.items()
    ]


def metrics_to_out(
    result: recon_bench.EvalResult,
) -> service_schemas.MetricsOut:
    """Convert an evaluation result's metrics to response models.

    Parameters
    ----------
    result
        Evaluation result returned by the core library.

    Returns
    -------
    service_schemas.MetricsOut
        JSON-safe grouped metric values.
    """
    return service_schemas.MetricsOut(
        image=tensor_metrics_to_out(result.image_metrics),
        geometry=tensor_metrics_to_out(result.geometry_metrics),
    )


def _profile_entry_to_dict(entry: object) -> dict[str, JsonValue]:
    if dataclasses.is_dataclass(entry):
        raw = {
            field.name: getattr(entry, field.name)
            for field in dataclasses.fields(entry)
        }
    else:
        raw = vars(entry)

    children = raw.get("children", [])
    data: dict[str, JsonValue] = {
        key: typing.cast(JsonValue, value)
        for key, value in raw.items()
        if key != "children"
    }
    data["children"] = [
        _profile_entry_to_dict(child)
        for child in typing.cast(list[object], children)
    ]
    return data


def profile_to_out(profile: object | None) -> service_schemas.ProfileOut | None:
    """Convert optional profiling output to response models.

    Parameters
    ----------
    profile
        ProfileResult-like object attached by the core evaluator.

    Returns
    -------
    service_schemas.ProfileOut or None
        JSON-safe profiling data, if profiling was enabled.
    """
    if profile is None:
        return None
    profile_result = typing.cast(recon_bench.ProfileResult, profile)
    return service_schemas.ProfileOut(
        timing=[
            _profile_entry_to_dict(entry)
            for entry in profile_result.timing
        ],
        memory=[
            _profile_entry_to_dict(entry)
            for entry in profile_result.memory
        ],
        cuda_available=bool(profile_result.cuda_available),
    )


def save_render_artifacts(
    result: recon_bench.EvalResult,
    *,
    job_id: str,
    config: service_config.ServiceConfig,
) -> list[ArtifactRecord]:
    """Save rendered images as PNG artifacts.

    Parameters
    ----------
    result
        Evaluation result that may contain rendered images.
    job_id
        UUID string for the owning job.
    config
        Runtime settings containing the storage root.

    Returns
    -------
    list[ArtifactRecord]
        Artifact response models paired with filesystem paths.

    Raises
    ------
    ValueError
        If the evaluator returns an unsupported render role.
    OSError
        If image artifact writing fails.
    """
    if result.rendered_images is None:
        return []

    saved: list[ArtifactRecord] = []
    for role, images in result.rendered_images.items():
        for index, image in enumerate(images):
            artifact_id = str(uuid.uuid4())
            output_path = service_storage.artifact_path(
                config,
                job_id,
                artifact_id,
            )
            io_image.save_image(image, output_path)
            saved.append(
                (
                    service_storage.artifact_out(
                        artifact_id=artifact_id,
                        role=role,
                        index=index,
                    ),
                    output_path,
                )
            )
    return saved


async def run_evaluate(
    semaphore: asyncio.Semaphore,
    kwargs: dict[str, object],
) -> recon_bench.EvalResult:
    """Run ``evaluate`` in a worker thread under the GPU semaphore.

    Parameters
    ----------
    semaphore
        Configurable concurrency guard for GPU-heavy evaluation work.
    kwargs
        Keyword arguments forwarded to ``recon_bench.evaluate``.

    Returns
    -------
    recon_bench.EvalResult
        Result from the core evaluator.

    Notes
    -----
    If the request task is cancelled, the worker thread cannot be stopped.
    This function waits for that thread to finish before releasing the
    semaphore so GPU concurrency limits remain intact.
    """
    async with semaphore:
        task = asyncio.create_task(
            asyncio.to_thread(recon_bench.evaluate, **kwargs),
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise
