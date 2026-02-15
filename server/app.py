"""FastAPI application for robot arm web control."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .events import EventBus
from .hardware import HardwareManager

logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize hardware on startup, clean up on shutdown."""
    settings: Settings = app.state.settings
    bus = EventBus()
    hw = HardwareManager(
        arm_port=settings.arm_port,
        sam2_model=settings.sam2_model,
        sam2_device=settings.sam2_device,
        calibration_path=settings.calibration_path,
        mock=settings.mock_hardware,
    )
    app.state.hardware = hw
    app.state.event_bus = bus

    await hw.start()
    logger.info("Hardware initialized")
    yield
    await hw.shutdown()
    logger.info("Hardware shut down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="RoArm Control Server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    from .routers.agent import router as agent_router
    from .routers.arm import router as arm_router
    from .routers.calibration import router as calibration_router
    from .routers.camera import router as camera_router
    from .routers.vision import router as vision_router

    app.include_router(arm_router)
    app.include_router(camera_router)
    app.include_router(vision_router)
    app.include_router(agent_router)
    app.include_router(calibration_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


def main():
    """Entry point for `uv run roarm-server`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
