"""ASGI application entrypoint for the optional service."""


def create_app():
    from fastapi import FastAPI

    app = FastAPI(title="recon-bench", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
