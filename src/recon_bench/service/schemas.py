"""Pydantic schemas for the optional service."""

from __future__ import annotations

import datetime
import enum
import typing

import pydantic

import recon_bench


MetricName = typing.Annotated[str, pydantic.Field(min_length=4, max_length=100)]
RgbInt = typing.Annotated[int, pydantic.Field(ge=0, le=255)]
UnitVectorComponent = typing.Annotated[float, pydantic.Field(ge=-1.0, le=1.0)]


class EvalMode(enum.StrEnum):
    IMAGE_VS_IMAGE = "image-vs-image"
    IMAGE_VS_MESH = "image-vs-mesh"
    MESH_VS_MESH = "mesh-vs-mesh"


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SAVING_ARTIFACTS = "saving_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"


class CameraIn(pydantic.BaseModel):
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[UnitVectorComponent, UnitVectorComponent, UnitVectorComponent] = (
        0.0,
        1.0,
        0.0,
    )
    fov: float = pydantic.Field(default=60.0, gt=0.0, lt=180.0)
    width: int = pydantic.Field(default=512, gt=0, lt=10000)
    height: int = pydantic.Field(default=512, gt=0, lt=10000)
    near: float = pydantic.Field(default=0.01, gt=0.0)
    far: float = pydantic.Field(default=100.0, gt=0.0)

    @pydantic.model_validator(mode="after")
    def validate_planes(self) -> CameraIn:
        if self.far <= self.near:
            raise ValueError("far must be greater than near")
        return self

    def to_camera(self) -> recon_bench.Camera:
        return recon_bench.Camera(**self.model_dump())


CameraList = typing.Annotated[list[CameraIn], pydantic.Field(min_length=1)]


class CamerasPayload(pydantic.BaseModel):
    camera: CameraIn | None = None
    cameras: CameraList | None = None

    @pydantic.model_validator(mode="after")
    def validate_one_camera_source(self) -> CamerasPayload:
        if self.camera is not None and self.cameras is not None:
            raise ValueError("provide either camera or cameras, not both")
        return self

    def camera_input(self) -> recon_bench.Camera | list[recon_bench.Camera] | None:
        if self.camera is not None:
            return self.camera.to_camera()
        if self.cameras is not None:
            return [camera.to_camera() for camera in self.cameras]
        return None


class EvalOptions(pydantic.BaseModel):
    metrics: list[MetricName] | None = None
    profile: bool = False
    save_renders: bool = False
    shard_size: int = pydantic.Field(default=10, gt=0)
    max_size: int | None = pydantic.Field(default=None, gt=0)
    background_color: tuple[RgbInt, RgbInt, RgbInt] = (255, 255, 255)

    def normalized_background_color(self) -> tuple[float, float, float]:
        return tuple(channel / 255.0 for channel in self.background_color)


class ImageVsImageOptions(EvalOptions):
    pass


class ImageVsMeshOptions(EvalOptions, CamerasPayload):
    @pydantic.model_validator(mode="after")
    def validate_camera_required(self) -> ImageVsMeshOptions:
        if self.camera is None and self.cameras is None:
            raise ValueError("image-vs-mesh requires camera or cameras")
        return self


class MeshVsMeshOptions(EvalOptions, CamerasPayload):
    image_eval: bool = False
    geometry_type: recon_bench.GeometryType = recon_bench.GeometryType.MESH
    num_points: int = pydantic.Field(default=10000, gt=0, le=10000000)

    @pydantic.model_validator(mode="after")
    def validate_image_eval_geometry(self) -> MeshVsMeshOptions:
        if self.image_eval and self.geometry_type == recon_bench.GeometryType.POINTCLOUD:
            raise ValueError("image_eval is only supported for mesh geometry")
        return self


class MetricValue(pydantic.BaseModel):
    name: str
    values: list[float]


class MetricsOut(pydantic.BaseModel):
    image: list[MetricValue] | None = None
    geometry: list[MetricValue] | None = None


class ArtifactOut(pydantic.BaseModel):
    artifact_id: str
    role: typing.Literal["target", "prediction"]
    index: int
    media_type: str
    url: str


class ProfileOut(pydantic.BaseModel):
    timing: list[dict[str, object]]
    memory: list[dict[str, object]]
    cuda_available: bool


class EvalResponse(pydantic.BaseModel):
    job_id: str
    status: JobStatus
    mode: EvalMode
    metrics: MetricsOut
    profile: ProfileOut | None = None
    artifacts: list[ArtifactOut] = pydantic.Field(default_factory=list)
    created_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class ErrorDetail(pydantic.BaseModel):
    code: str
    message: str
    details: dict[str, object] = pydantic.Field(default_factory=dict)
    job_id: str | None = None


class ErrorResponse(pydantic.BaseModel):
    error: ErrorDetail
