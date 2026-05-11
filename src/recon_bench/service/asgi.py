"""ASGI application entrypoint for the optional service."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI

from .config import get_config
from .db import Database
from .routes import router
from .storage import ensure_service_dirs


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    ensure_service_dirs(config)
    database = Database(config.db_path)
    await database.init()
    app.state.config = config
    app.state.db = database
    app.state.gpu_semaphore = asyncio.Semaphore(config.gpu_slots)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="recon-bench", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
