"""ASGI application entrypoint for the optional service."""

import asyncio
import contextlib
import logging
import typing

import fastapi
import fastapi.openapi.utils
import pydantic

import recon_bench.service.config as service_config
import recon_bench.service.db as service_db
import recon_bench.service.routes as service_routes
import recon_bench.service.schemas as service_schemas
import recon_bench.service.storage as service_storage


LOGGER: logging.Logger = logging.getLogger(__name__)
SERVICE_TITLE: str = "recon-bench"
SERVICE_VERSION: str = "0.1.0"
MULTIPART_CONTENT_TYPE: str = "multipart/form-data"
JSON_CONTENT_TYPE: str = "application/json"
OPTIONS_FORM_FIELD: str = "options"
SCHEMA_REF_PREFIX: str = "#/components/schemas/"
OpenApiDocument = dict[str, object]
OPENAPI_OPTION_MODELS: dict[str, type[pydantic.BaseModel]] = {
    f"{service_routes.EVALS_PREFIX}/image-vs-image": (
        service_schemas.ImageVsImageOptions
    ),
    f"{service_routes.EVALS_PREFIX}/image-vs-mesh": (
        service_schemas.ImageVsMeshOptions
    ),
    f"{service_routes.EVALS_PREFIX}/mesh-vs-mesh": (
        service_schemas.MeshVsMeshOptions
    ),
}


def _object_map(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return typing.cast(dict[str, object], value)
    return None


def _component_schemas(document: OpenApiDocument) -> dict[str, object]:
    components = document.setdefault("components", {})
    if not isinstance(components, dict):
        components = {}
        document["components"] = components

    schemas = components.setdefault("schemas", {})
    if isinstance(schemas, dict):
        return schemas

    components["schemas"] = {}
    return typing.cast(dict[str, object], components["schemas"])


def _register_service_components(document: OpenApiDocument) -> None:
    schemas = _component_schemas(document)
    for name, schema in service_schemas.openapi_components().items():
        schemas.setdefault(name, schema)


def _schema_properties(schema: dict[str, object]) -> dict[str, object]:
    properties = schema.setdefault("properties", {})
    if isinstance(properties, dict):
        return typing.cast(dict[str, object], properties)

    schema["properties"] = {}
    return typing.cast(dict[str, object], schema["properties"])


def _multipart_schema(
    document: OpenApiDocument,
    path: str,
) -> dict[str, object]:
    paths = _object_map(document.get("paths")) or {}
    path_item = _object_map(paths.get(path)) or {}
    operation = _object_map(path_item.get("post")) or {}
    request_body = _object_map(operation.get("requestBody")) or {}
    content = _object_map(request_body.get("content")) or {}
    multipart = _object_map(content.get(MULTIPART_CONTENT_TYPE)) or {}

    encoding = multipart.setdefault("encoding", {})
    if isinstance(encoding, dict):
        encoding[OPTIONS_FORM_FIELD] = {"contentType": JSON_CONTENT_TYPE}

    return _object_map(multipart.get("schema")) or {}


def _resolve_schema_ref(
    document: OpenApiDocument,
    schema: dict[str, object],
) -> dict[str, object]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith(SCHEMA_REF_PREFIX):
        return schema

    component_name = ref.removeprefix(SCHEMA_REF_PREFIX)
    component = _component_schemas(document).get(component_name)
    return _object_map(component) or schema


def _patch_multipart_option_schemas(document: OpenApiDocument) -> None:
    for path, model in OPENAPI_OPTION_MODELS.items():
        schema = _resolve_schema_ref(
            document,
            _multipart_schema(document, path),
        )
        properties = _schema_properties(schema)
        properties[OPTIONS_FORM_FIELD] = {
            "$ref": f"{SCHEMA_REF_PREFIX}{model.__name__}",
        }


def _install_openapi(app: fastapi.FastAPI) -> None:
    """Install service-aware OpenAPI generation.

    Parameters
    ----------
    app
        FastAPI application whose OpenAPI schema should include service
        multipart option models.
    """

    def custom_openapi() -> OpenApiDocument:
        if app.openapi_schema is not None:
            return typing.cast(OpenApiDocument, app.openapi_schema)

        document = fastapi.openapi.utils.get_openapi(
            title=SERVICE_TITLE,
            version=SERVICE_VERSION,
            routes=app.routes,
        )
        _register_service_components(document)
        _patch_multipart_option_schemas(document)
        app.openapi_schema = document
        return document

    app.openapi = custom_openapi


@contextlib.asynccontextmanager
async def lifespan(
    app: fastapi.FastAPI,
) -> typing.AsyncIterator[None]:
    """Initialize service state for the ASGI lifespan.

    Parameters
    ----------
    app
        FastAPI application receiving configured service state.

    Yields
    ------
    None
        Control to FastAPI while the application is serving requests.
    """
    config = service_config.get_config()
    service_storage.ensure_service_dirs(config)
    database = service_db.Database(config.db_path)
    await database.init()
    app.state.config = config
    app.state.db = database
    app.state.gpu_semaphore = asyncio.Semaphore(config.gpu_slots)
    LOGGER.info("service startup complete")
    yield
    LOGGER.info("service shutdown complete")


def create_app() -> fastapi.FastAPI:
    """Create the FastAPI application.

    Returns
    -------
    fastapi.FastAPI
        Configured ASGI app with service routes registered.
    """
    app = fastapi.FastAPI(
        title=SERVICE_TITLE,
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.include_router(service_routes.router)
    _install_openapi(app)
    return app


app = create_app()
