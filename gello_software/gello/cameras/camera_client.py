"""ZMQ client for the camera server.

Used by eval launchers to pull the latest 3-camera observation without
holding RealSense devices in-process. See ``camera_server.py`` for the wire
protocol.
"""
from __future__ import annotations

import logging
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import zmq


logger = logging.getLogger(__name__)


class CameraClientError(RuntimeError):
    """Raised when the camera server is unreachable, slow, or returns an error."""


class CameraClient:
    """REQ-side wrapper. ``get_obs()`` returns ``{cam_name: np.ndarray (H,W,3) uint8 RGB}``.

    On timeout the underlying REQ socket is closed and recreated — REQ sockets
    become unusable after a recv timeout without a matching reply.
    """

    def __init__(
        self,
        endpoint: str,
        request_timeout_ms: int = 500,
        max_frame_age_sec: Optional[float] = 0.5,
    ) -> None:
        self.endpoint = endpoint
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_frame_age_sec = max_frame_age_sec
        self._ctx = zmq.Context.instance()
        self._sock: Optional[zmq.Socket] = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        sock.connect(self.endpoint)
        self._sock = sock

    def _request(self, cmd: str) -> Dict[str, Any]:
        assert self._sock is not None
        try:
            self._sock.send(pickle.dumps({"cmd": cmd}))
            raw = self._sock.recv()
        except zmq.Again as exc:
            # REQ socket is now in a bad state; reset before raising.
            self._connect()
            raise CameraClientError(
                f"Camera server timeout ({self.request_timeout_ms} ms) on cmd={cmd!r} "
                f"at {self.endpoint}. Is the server running?"
            ) from exc
        try:
            resp = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 — unparseable reply
            self._connect()
            raise CameraClientError(f"Unparseable reply from camera server: {exc!r}") from exc
        if not resp.get("ok"):
            raise CameraClientError(f"Server error: {resp.get('error')}")
        return resp

    def ping(self) -> bool:
        return bool(self._request("ping").get("pong"))

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Return ``{cam_name: np.ndarray (H,W,3) uint8 RGB}`` with the latest frames."""
        resp = self._request("obs")
        frames: Dict[str, np.ndarray] = resp["frames"]
        if self.max_frame_age_sec is not None:
            now = time.time()
            for name, ts in (resp.get("timestamps") or {}).items():
                if ts and (now - ts) > self.max_frame_age_sec:
                    raise CameraClientError(
                        f"Stale frame from {name}: {now - ts:.3f}s old "
                        f"(>{self.max_frame_age_sec:.3f}s)."
                    )
        return frames

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            self._sock = None


class CameraSubscriber:
    """Optional PUB/SUB consumer for the live viewer.

    The eval inner loop should use ``CameraClient`` (REQ/REP). This subscriber
    exists so a cv2 viewer can render at camera rate without competing for the
    REP socket with the policy.
    """

    def __init__(self, endpoint: str, recv_timeout_ms: int = 100) -> None:
        self.endpoint = endpoint
        self.recv_timeout_ms = int(recv_timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(endpoint)

    def try_recv(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the most recent frame dict if one is available, else None."""
        latest: Optional[bytes] = None
        # Drain the queue so we hand the consumer the freshest frame.
        while True:
            try:
                latest = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return None
        try:
            resp = pickle.loads(latest)
        except Exception as exc:  # noqa: BLE001 — drop malformed publish
            logger.warning("Dropped malformed PUB payload: %s", exc)
            return None
        if not resp.get("ok"):
            return None
        return resp.get("frames")

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
