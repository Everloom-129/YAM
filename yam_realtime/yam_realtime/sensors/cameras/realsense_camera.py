import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from yam_realtime.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    return [dev.get_info(rs.camera_info.serial_number) for dev in devices]


@dataclass
class RealSenseCamera(CameraDriver):
    """
    RealSense camera driver with background frame capture.
    """

    device_id: Optional[str] = None
    camera_type: str = "realsense_camera"
    image_transfer_time_offset: int = 80
    resolution: Tuple[int, int] = (640, 360)  # (width, height)
    fps: int = 30
    flip: bool = False
    name: Optional[str] = None

    def __post_init__(self):
        import pyrealsense2 as rs

        self._rs = rs
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread = None

        self._warmup_frames = 15
        self._read_timeout_ms = 1200
        self._read_wait_timeout_sec = 1.5
        self._max_frame_age_sec = 0.30

        self._latest_color_image = None
        self._latest_depth_image = None
        self._latest_frame_timestamp = None
        self._last_capture_error = None

        self._align = rs.align(rs.stream.color)
        self._pipeline = None
        self._config = None

        self._start_pipeline()
        self._start_capture_thread()

    def __repr__(self) -> str:
        return (
            "RealSenseCamera("
            f"device_id={self.device_id!r}, name={self.name!r}, "
            f"resolution={self.resolution}, fps={self.fps}, flip={self.flip})"
        )

    def _start_capture_thread(self) -> None:
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"realsense_capture_{self.device_id or self.name or 'default'}",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    frames = self._pipeline.wait_for_frames(timeout_ms=self._read_timeout_ms)
                    frames = self._align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    raise RuntimeError("Invalid RealSense frame pair received.")

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                timestamp = time.time()

                with self._frame_lock:
                    self._latest_color_image = color_image
                    self._latest_depth_image = depth_image
                    self._latest_frame_timestamp = timestamp
                    self._last_capture_error = None
                    self._frame_ready.set()

                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                with self._frame_lock:
                    self._last_capture_error = exc
                if consecutive_failures >= 5:
                    self._frame_ready.set()
                time.sleep(0.05)
                self._start_pipeline()

    def _start_pipeline(self) -> None:
        rs = self._rs
        width, height = self.resolution

        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass

            self._pipeline = rs.pipeline()
            self._config = rs.config()

            if self.device_id:
                self._config.enable_device(self.device_id)

            self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)
            self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
            self._pipeline.start(self._config)

            for _ in range(self._warmup_frames):
                self._pipeline.wait_for_frames()

    def read(self) -> CameraData:
        import cv2

        if not self._frame_ready.wait(timeout=self._read_wait_timeout_sec):
            raise RuntimeError("Timed out waiting for RealSense capture thread to produce a frame.")

        with self._frame_lock:
            color_image = self._latest_color_image
            depth_image = self._latest_depth_image
            frame_timestamp = self._latest_frame_timestamp
            last_error = self._last_capture_error

        if color_image is None or depth_image is None or frame_timestamp is None:
            if last_error is not None:
                raise RuntimeError("RealSense capture thread failed to produce a frame.") from last_error
            raise RuntimeError("RealSense frame is unavailable.")

        frame_age = time.time() - frame_timestamp
        if frame_age > self._max_frame_age_sec:
            raise RuntimeError(f"RealSense frame is stale ({frame_age:.3f}s old); camera may be stalled.")

        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth = depth_image

        if self.flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)

        timestamp_ms = frame_timestamp * 1000 - self.image_transfer_time_offset
        return CameraData(images={"rgb": image, "depth": depth[:, :, None]}, timestamp=timestamp_ms)

    def get_camera_info(self) -> Dict[str, Any]:
        return {
            "camera_type": self.camera_type,
            "device_id": self.device_id,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
            "flip": self.flip,
            "name": self.name,
        }

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")
