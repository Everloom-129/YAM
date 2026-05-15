import atexit
import os
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.env import RobotEnv
from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.utils.logging_utils import log_collect_demos
from molmoact import MolmoAct

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

# Global variables for cleanup
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None
_video_recorders: list = []


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    for recorder in _video_recorders:
        recorder.stop()
    if _bimanual:
        move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
    else:
        move_to_start_position(_env, _bimanual, _left_cfg)
    print("Cleanup completed.")


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    video_output_dir: Optional[str] = None
    """Directory to write the recorded video files. Defaults to <storage.base_dir>/videos."""

    video_fps: int = 30
    """Target frames-per-second for the recorded videos."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    os._exit(0)


class VideoRecorder:
    """Background thread that pulls frames from a camera and pipes them to ffmpeg.

    Writes a standard H.264 MP4 (yuv420p) so the file plays natively in Ubuntu's
    default Videos/Totem app and most other players without extra codec packs.
    """

    def __init__(self, camera: RealSenseCamera, output_path: str, fps: int = 30):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install it with `sudo apt install ffmpeg`."
            )
        self._camera = camera
        self._output_path = output_path
        self._fps = fps
        self._period = 1.0 / float(fps)
        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._record_loop,
            name=f"video_recorder_{os.path.basename(output_path)}",
            daemon=True,
        )
        self._frame_count = 0

    def start(self):
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        self._thread.start()
        logger.info(f"Started video recording -> {self._output_path}")

    def _ensure_proc(self, frame_shape):
        if self._proc is not None:
            return
        height, width = frame_shape[:2]
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(self._fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self._output_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _record_loop(self):
        next_frame_time = time.time()
        while not self._stop_event.is_set():
            try:
                rgb_image, _ = self._camera.read()
                self._ensure_proc(rgb_image.shape)
                # camera.read() returns RGB; ffmpeg input pix_fmt is bgr24
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                self._proc.stdin.write(bgr_image.tobytes())
                self._frame_count += 1
            except (BrokenPipeError, ValueError):
                # ffmpeg exited; stop recording.
                break
            except Exception as exc:
                logger.warning(f"Video recorder dropped a frame: {exc}")

            next_frame_time += self._period
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Reset cadence if we fell behind to avoid runaway catch-up.
                next_frame_time = time.time()

    def stop(self):
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        logger.info(
            f"Stopped video recording ({self._frame_count} frames written) -> {self._output_path}"
        )


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    # left, right front camera (the device id order is based on the plugged in order on the adapter)
    ids = get_device_ids()
    print(f"Found {len(ids)} camera devices")
    print(ids)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )

    camera_cfg = left_cfg["sensors"]["cameras"]
    cameras = {
        "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
        "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
        "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
    }

    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    env = RobotEnv(robot, control_rate_hz=cfg.get("hz", 30), camera_dict=cameras)

    # Store global variables for cleanup
    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None

    # Move robot to start_joints position if specified in config
    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(f"Launching robot: {robot.__class__.__name__}")
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    # Set up a video recorder for each camera.
    video_output_dir = args.video_output_dir or os.path.join(
        left_cfg["storage"]["base_dir"], "videos"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = left_cfg["storage"].get("task_directory", "eval")

    global _video_recorders
    for camera_name, camera in cameras.items():
        video_path = os.path.join(
            video_output_dir, f"{task_dir}_{camera_name}_{timestamp}.mp4"
        )
        recorder = VideoRecorder(
            camera=camera,
            output_path=video_path,
            fps=args.video_fps,
        )
        recorder.start()
        _video_recorders.append(recorder)

    molmoact = MolmoAct()
    try:
        run_control_loop_eval(
            env,
            policy=molmoact,
            instruction=left_cfg["storage"]["language_instruction"],
        )
    finally:
        for recorder in _video_recorders:
            recorder.stop()


def run_control_loop_eval(
    env: RobotEnv,
    policy: MolmoAct = None,
    instruction: str = None,
) -> None:
    """Run the main control loop."""
    obs = env.get_obs()
    logger.info("Starting policy inference...")

    while True:
        log_collect_demos("Running policy inference...", "info")
        input_dict = policy.prepare_input(obs, instruction)
        start_time = time.time()
        actions = policy.inference(input_dict)["actions"]
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")
        for i in range(len(actions)):
            obs = dynamic_smoothing(env, np.array(actions[i]))


def smooth_move_while_inference_envstep(env: RobotEnv, action):
    current_joint = env.get_obs()["joint_positions"]
    target_joint = action

    steps = 5
    obs = None
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        interpolated_joint = (1 - alpha) * current_joint + alpha * target_joint  # Linear interpolation
        obs = env.step(interpolated_joint)
        time.sleep(0.5 / steps)

    return obs


def dynamic_smoothing(
    env,
    target_joints: np.ndarray,
):
    curr_joints = env.get_obs()["joint_positions"]

    max_delta = (np.abs(curr_joints - target_joints)).max()
    steps = min(int(max_delta / 0.01), 100)
    if steps <= 1:
        obs = env.step(target_joints)
        return obs
    print(f"Moving to start position with {steps} steps")

    obs = None
    for jnt in np.linspace(curr_joints, target_joints, steps):
        obs = env.step(jnt)
        time.sleep(0.001)
    return obs


if __name__ == "__main__":
    main()
