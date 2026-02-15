"""Agent endpoints: task submission, cancellation, confirmation, and WebSocket events."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..agent_wrapper import WebAgentV2
from ..config import Settings
from ..events import EventBus
from ..hardware import HardwareManager

logger = logging.getLogger("server.routers.agent")
router = APIRouter(tags=["agent"])

# Singleton agent (created on first task submission)
_agent: WebAgentV2 | None = None


def _get_or_create_agent(request: Request) -> WebAgentV2:
    global _agent
    if _agent is None:
        hw: HardwareManager = request.app.state.hardware
        bus: EventBus = request.app.state.event_bus
        settings: Settings = request.app.state.settings
        _agent = WebAgentV2(
            hw=hw,
            bus=bus,
            openai_api_key=settings.openai_api_key,
            helicone_api_key=settings.helicone_api_key,
            model=settings.llm_model,
            reasoning_effort=settings.reasoning_effort,
        )
    return _agent


class TaskRequest(BaseModel):
    task: str
    auto_confirm: bool = False


class ConfirmRequest(BaseModel):
    approved: bool


@router.post("/api/agent/task")
async def submit_task(body: TaskRequest, request: Request):
    agent = _get_or_create_agent(request)
    try:
        task_id = await agent.submit_task(body.task, body.auto_confirm)
        return {"task_id": task_id, "status": "started"}
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/api/agent/cancel")
async def cancel_task(request: Request):
    agent = _get_or_create_agent(request)
    await agent.cancel()
    return {"status": "cancelling"}


@router.post("/api/agent/confirm")
async def confirm_action(body: ConfirmRequest, request: Request):
    agent = _get_or_create_agent(request)
    await agent.confirm_action(body.approved)
    return {"status": "confirmed" if body.approved else "rejected"}


@router.get("/api/agent/status")
async def agent_status(request: Request):
    global _agent
    if _agent is None:
        return {"state": "idle", "task": None}
    return {"state": _agent.state, "task": _agent.current_task}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time events."""
    bus: EventBus = websocket.app.state.event_bus
    await websocket.accept()
    await bus.subscribe(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await bus.unsubscribe(websocket)
