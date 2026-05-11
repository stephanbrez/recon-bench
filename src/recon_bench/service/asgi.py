"""ASGI application entrypoint for the optional service."""

import asyncio
import contextlib
import logging
import typing

import fastapi

import recon_bench.service.config as service_config
import recon_bench.service.db as service_db
import recon_bench.service.routes as service_routes
import recon_bench.service.storage as service_storage


LOGGER: logging.Logger = logging.getLogger(__name__)
SERVICE_TITLE: str = "recon-bench"
SERVICE_VERSION: str = "0.1.0"


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
    return app


app = create_app()
