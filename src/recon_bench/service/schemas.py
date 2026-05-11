"""Pydantic schemas for the optional service."""

import datetime
import enum
import typing

import pydantic

import recon_bench


MetricName = typing.Annotated[str, pydantic.Field(min_length=4, max_length=100)]
RgbInt = typing.Annotated[int, pydantic.Field(ge=0, le=255)]
UnitVectorComponent = typing.Annotated[float, pydantic.Field(ge=-1.0, le=1.0)]
ArtifactRole = typing.Literal["target", "prediction"]

ARTIFACT_ROLES: frozenset[str] = frozenset({"target", "prediction"})


class EvalMode(enum.StrEnum):
    """Supported service evaluation modes."""

    IMAGE_VS_IMAGE = "image-vs-image"
    IMAGE_VS_MESH = "image-vs-mesh"
    MESH_VS_MESH = "mesh-vs-mesh"


class JobStatus(enum.StrEnum):
    """Persisted lifecycle state for a service job."""

    PENDING = "pending"
    RUNNING = "running"
    SAVING_ARTIFACTS = "saving_artifacts"
    COMPLETED = "completed"
    FAILED = "failed"


class CameraIn(pydantic.BaseModel):
    """Request model for a render camera.

    Parameters
    ----------
    position
        Camera position in world coordinates.
    look_at
        World-space point the camera points toward.
    up
        Unit vector components for world-space up.
    fov
        Vertical field of view in degrees.
    width
        Render width in pixels.
    height
        Render height in pixels.
    near
        Near clipping plane distance.
    far
        Far clipping plane distance.
    """

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
    def validate_planes(self) -> "CameraIn":
        if self.far <= self.near:
            raise ValueError("far must be greater than near")
        return self

    def to_camera(self) -> recon_bench.Camera:
        return recon_bench.Camera(**self.model_dump())


CameraList = typing.Annotated[list[CameraIn], pydantic.Field(min_length=1)]


class CamerasPayload(pydantic.BaseModel):
    """Request mixin for either one camera or a list of cameras."""

    camera: CameraIn | None = None
    cameras: CameraList | None = None

    @pydantic.model_validator(mode="after")
    def validate_one_camera_source(self) -> "CamerasPayload":
        if self.camera is not None and self.cameras is not None:
            raise ValueError("provide either camera or cameras, not both")
        return self

    def camera_input(
        self,
    ) -> recon_bench.Camera | list[recon_bench.Camera] | None:
        if self.camera is not None:
            return self.camera.to_camera()
        if self.cameras is not None:
            return [camera.to_camera() for camera in self.cameras]
        return None


class EvalOptions(pydantic.BaseModel):
    """Common evaluator options shared by all service modes."""

    metrics: list[MetricName] | None = None
    profile: bool = False
    save_renders: bool = False
    shard_size: int = pydantic.Field(default=10, gt=0)
    max_size: int | None = pydantic.Field(default=None, gt=0)
    background_color: tuple[RgbInt, RgbInt, RgbInt] = (255, 255, 255)

    def normalized_background_color(self) -> tuple[float, float, float]:
        return tuple(channel / 255.0 for channel in self.background_color)


class ImageVsImageOptions(EvalOptions):
    """Options for image-vs-image evaluation requests."""

    pass


class ImageVsMeshOptions(EvalOptions, CamerasPayload):
    """Options for image-vs-mesh evaluation requests."""

    @pydantic.model_validator(mode="after")
    def validate_camera_required(self) -> "ImageVsMeshOptions":
        if self.camera is None and self.cameras is None:
            raise ValueError("image-vs-mesh requires camera or cameras")
        return self


class MeshVsMeshOptions(EvalOptions, CamerasPayload):
    """Options for mesh-vs-mesh evaluation requests."""

    image_eval: bool = False
    geometry_type: recon_bench.GeometryType = recon_bench.GeometryType.MESH
    num_points: int = pydantic.Field(default=10000, gt=0, le=10000000)

    @pydantic.model_validator(mode="after")
    def validate_image_eval_geometry(self) -> "MeshVsMeshOptions":
        if (
            self.image_eval
            and self.geometry_type == recon_bench.GeometryType.POINTCLOUD
        ):
            raise ValueError("image_eval is only supported for mesh geometry")
        return self


class MetricValue(pydantic.BaseModel):
    """Per-item values for one metric."""

    name: str
    values: list[float]


class MetricsOut(pydantic.BaseModel):
    """Grouped metric output from an evaluation."""

    image: list[MetricValue] | None = None
    geometry: list[MetricValue] | None = None


class ArtifactOut(pydantic.BaseModel):
    """Metadata for a generated artifact."""

    artifact_id: str
    role: ArtifactRole
    index: int
    media_type: str
    url: str


class ProfileOut(pydantic.BaseModel):
    """JSON-safe profiling output."""

    timing: list[dict[str, object]]
    memory: list[dict[str, object]]
    cuda_available: bool


class EvalResponse(pydantic.BaseModel):
    """Final response returned by evaluation endpoints."""

    job_id: str
    status: JobStatus
    mode: EvalMode
    metrics: MetricsOut
    profile: ProfileOut | None = None
    artifacts: list[ArtifactOut] = pydantic.Field(default_factory=list)
    created_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class ErrorDetail(pydantic.BaseModel):
    """Structured error body for service failures."""

    code: str
    message: str
    details: dict[str, object] = pydantic.Field(default_factory=dict)
    job_id: str | None = None


class ErrorResponse(pydantic.BaseModel):
    """Top-level error response wrapper."""

    error: ErrorDetail
