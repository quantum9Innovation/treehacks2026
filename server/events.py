"""WebSocket pub/sub event bus for real-time updates."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket

logger = logging.getLogger("server.events")


class EventBus:
    """Broadcasts typed events to all connected WebSocket clients."""

    def __init__(self):
        self._subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket):
        async with self._lock:
            self._subscribers.add(ws)
        logger.info(f"WebSocket subscribed, total={len(self._subscribers)}")

    async def unsubscribe(self, ws: WebSocket):
        async with self._lock:
            self._subscribers.discard(ws)
        logger.info(f"WebSocket unsubscribed, total={len(self._subscribers)}")

    async def publish(self, event_type: str, payload: dict):
        """Publish an event to all subscribers."""
        message = json.dumps(
            {
                "type": event_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        async with self._lock:
            dead: set[WebSocket] = set()
            for ws in self._subscribers:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            self._subscribers -= dead

    def publish_sync(self, event_type: str, payload: dict):
        """Thread-safe publish from synchronous context (e.g. agent loop thread)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.publish(event_type, payload), loop
                )
            else:
                loop.run_until_complete(self.publish(event_type, payload))
        except RuntimeError:
            logger.warning(f"Could not publish event {event_type}: no event loop")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
