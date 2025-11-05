from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

import asyncio

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..person_following.config import FollowerConfig
from ..person_following.pipeline import PersonFollowerPipeline, PipelineState
from ..person_following.camera import RealSenseUnavailableError

LOG = logging.getLogger(__name__)

app = FastAPI(title="Person Follower UI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: PersonFollowerPipeline | None = None
_pipeline_error: str | None = None


@app.on_event("startup")
def startup_pipeline() -> None:
    global _pipeline, _pipeline_error

    config = FollowerConfig()

    model_env = os.environ.get("FOLLOWER_MODEL")
    if model_env:
        config.detector.model_name = model_env
    device_env = os.environ.get("FOLLOWER_DEVICE")
    half_env = os.environ.get("FOLLOWER_HALF")

    if device_env:
        config.detector.device = device_env
    else:
        try:
            import torch  # type: ignore[import]
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            config.detector.device = "cuda:0"
            if platform.machine().lower() == "aarch64":
                config.detector.use_half_precision = True

    if half_env is not None:
        config.detector.use_half_precision = half_env.lower() in {"1", "true", "yes", "on"}
    elif config.detector.use_half_precision and (
        config.detector.device is None or config.detector.device.startswith("cpu")
    ):
        LOG.warning("Disabling half precision for web pipeline because no CUDA device is configured.")
        config.detector.use_half_precision = False

    try:
        pipeline = PersonFollowerPipeline(config)
    except RealSenseUnavailableError as exc:
        LOG.error("Failed to initialize RealSense pipeline: %s", exc)
        _pipeline = None
        _pipeline_error = str(exc)
        return

    _pipeline = pipeline
    _pipeline_error = None
    pipeline.start()
    LOG.info(
        "Person follower pipeline started for web UI (device=%s, half=%s)",
        config.detector.device or "auto",
        config.detector.use_half_precision,
    )


@app.on_event("shutdown")
def shutdown_pipeline() -> None:
    if _pipeline:
        _pipeline.stop()
        LOG.info("Person follower pipeline stopped")


@app.get("/api/health")
def health() -> dict[str, Any]:
    if _pipeline_error:
        return {"status": "error", "message": _pipeline_error}
    if _pipeline is None:
        return {"status": "initializing"}
    pipeline_error = _pipeline.error()
    if pipeline_error:
        return {"status": "error", "message": pipeline_error}
    state = _pipeline.latest_state()
    return {"status": "ok" if state else "warming_up"}


@app.get("/api/status")
def status() -> JSONResponse:
    pipeline = _require_pipeline()

    state = pipeline.latest_state()
    if state is None:
        return JSONResponse({"status": "warming_up"})

    payload = {
        "status": "ok",
        "timestamp": state.timestamp,
        "detection": _serialize_detection(state),
        "estimate": _serialize_estimate(state),
        "command": _serialize_command(state),
        "fps": state.fps,
    }
    return JSONResponse(payload)


@app.get("/api/frame")
def frame() -> Response:
    pipeline = _require_pipeline()

    state = pipeline.latest_state()
    if state is None or state.annotated_frame_jpeg is None:
        raise HTTPException(status_code=404, detail="Frame not available yet.")

    headers = {"Cache-Control": "no-store, no-cache, must-revalidate"}
    return Response(content=state.annotated_frame_jpeg, media_type="image/jpeg", headers=headers)


@app.get("/api/frame/stream")
async def frame_stream() -> StreamingResponse:
    pipeline = _require_pipeline()

    async def generator():
        boundary = b"--frame"
        while True:
            state = pipeline.latest_state()
            if state is None or state.annotated_frame_jpeg is None:
                await asyncio.sleep(0.05)
                continue
            payload = (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(state.annotated_frame_jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + state.annotated_frame_jpeg
                + b"\r\n"
            )
            yield payload
            await asyncio.sleep(0.05)

    headers = {"Cache-Control": "no-store, no-cache, must-revalidate"}
    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )

def _serialize_detection(state: PipelineState) -> dict[str, Any] | None:
    detection = state.detection
    if detection is None:
        return None
    return {
        "bbox": detection.xyxy,
        "confidence": detection.confidence,
        "center": detection.center,
        "size": {"width": detection.width, "height": detection.height},
    }


def _serialize_estimate(state: PipelineState) -> dict[str, Any] | None:
    estimate = state.estimate
    if estimate is None:
        return None
    return {
        "distance_m": estimate.distance_m,
        "bearing_rad": estimate.bearing_rad,
        "bearing_deg": estimate.bearing_deg,
        "elevation_rad": estimate.elevation_rad,
        "elevation_deg": estimate.elevation_deg,
        "position_m": estimate.position_m,
        "valid_depth_ratio": estimate.valid_depth_ratio,
    }


def _serialize_command(state: PipelineState) -> dict[str, Any]:
    command = state.command
    return {
        "linear_velocity_mps": command.linear_velocity_mps,
        "angular_velocity_radps": command.angular_velocity_radps,
        "distance_error_m": command.distance_error_m,
        "angle_error_rad": command.angle_error_rad,
    }


def _mount_static(app: FastAPI) -> None:
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def _require_pipeline() -> PersonFollowerPipeline:
    if _pipeline_error:
        raise HTTPException(status_code=503, detail=_pipeline_error)
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not running.")
    pipeline_error = _pipeline.error()
    if pipeline_error:
        raise HTTPException(status_code=503, detail=pipeline_error)
    return _pipeline


_mount_static(app)
