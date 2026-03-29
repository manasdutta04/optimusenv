"""OpenEnv scaffold entrypoint for the OptimusEnv FastAPI server."""

from __future__ import annotations

from app.main import app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
