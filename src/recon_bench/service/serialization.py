"""JSON-safe conversion helpers for evaluation results."""

from __future__ import annotations

import asyncio
import typing
import uuid

import torch

import recon_bench
from recon_bench.io import image as io_image

from .config import ServiceConfig
from .schemas import ArtifactOut, MetricsOut, MetricValue, ProfileOut
from .storage import artifact_out, artifact_path


def tensor_metrics_to_out(metrics: dict[str, torch.Tensor] | None) -> list[MetricValue] | None:
    if metrics is None:
        return None
    return [
        MetricValue(
            name=name,
            values=[float(value) for value in tensor.detach().cpu().flatten().tolist()],
        )
        for name, tensor in metrics.items()
    ]


def metrics_to_out(result: recon_bench.EvalResult) -> MetricsOut:
    return MetricsOut(
        image=tensor_metrics_to_out(result.image_metrics),
        geometry=tensor_metrics_to_out(result.geometry_metrics),
    )


def _profile_entry_to_dict(entry: object) -> dict[str, object]:
    data = {
        key: value
        for key, value in vars(entry).items()
        if key != "children"
    }
    data["children"] = [
        _profile_entry_to_dict(child)
        for child in getattr(entry, "children", [])
    ]
    return data


def profile_to_out(profile: object | None) -> ProfileOut | None:
    if profile is None:
        return None
    return ProfileOut(
        timing=[_profile_entry_to_dict(entry) for entry in profile.timing],
        memory=[_profile_entry_to_dict(entry) for entry in profile.memory],
        cuda_available=bool(profile.cuda_available),
    )


def save_render_artifacts(
    result: recon_bench.EvalResult,
    *,
    job_id: str,
    config: ServiceConfig,
) -> list[tuple[ArtifactOut, typing.Any]]:
    if result.rendered_images is None:
        return []

    saved: list[tuple[ArtifactOut, typing.Any]] = []
    for role, images in result.rendered_images.items():
        for index, image in enumerate(images):
            artifact_id = str(uuid.uuid4())
            output_path = artifact_path(config, job_id, artifact_id)
            io_image.save_image(image, output_path)
            saved.append((artifact_out(artifact_id=artifact_id, role=role, index=index), output_path))
    return saved


async def run_evaluate(
    semaphore: asyncio.Semaphore,
    **kwargs: typing.Any,
) -> recon_bench.EvalResult:
    async with semaphore:
        return await asyncio.to_thread(recon_bench.evaluate, **kwargs)
