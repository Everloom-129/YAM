import atexit
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _ensure_lerobot_import() -> None:
    """Make local lerobot/src importable when running from this repo."""
    repo_root = Path(__file__).resolve().parents[3]
    lerobot_src = repo_root / "lerobot" / "src"
    if str(lerobot_src) not in sys.path and lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))


_ensure_lerobot_import()
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


LOGGER = logging.getLogger("fast_lerobot_saver")
LOGGER.setLevel(logging.INFO)


STATE_DIM_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
]


class FastLeRobotDataSaver:
    """
    High-throughput direct LeRobot v3.0 saver with resume support.

    Designed as a replacement for json-first pipelines:
      - Collect observations in RAM for one episode via add_observation()
      - Persist one episode atomically via save_episode()
      - Resume interrupted runs by re-opening an existing dataset directory
      - Finalize and optionally push to Hugging Face Hub via close()

    Expected observation keys (same shape as current collection loop):
      - joint_positions: (14,)
      - next_joint: (14,) [optional, falls back to joint_positions]
      - left_camera_rgb, right_camera_rgb, front_camera_rgb: HxWx3 uint8 [optional per camera]
    """

    def __init__(
        self,
        save_dir: str,
        task_directory: str,
        repo_id: str,
        language_instruction: str = "perform the task",
        fps: int = 30,
        robot_type: str = "molmoact_dual_arm",
        resume: bool = True,
        vcodec: str = "h264",
        batch_encoding_size: int = 8,
        image_writer_processes: int = 0,
        image_writer_threads: int = 12,
    ):
        self.root = Path(save_dir) / task_directory
        self.repo_id = repo_id
        self.language_instruction = language_instruction
        self.fps = int(fps)
        self.robot_type = robot_type
        self.resume = bool(resume)
        self.vcodec = vcodec
        self.batch_encoding_size = int(batch_encoding_size)
        self.image_writer_processes = int(image_writer_processes)
        self.image_writer_threads = int(image_writer_threads)

        self.buffer: List[Dict[str, Any]] = []
        self._dataset: Optional[LeRobotDataset] = None
        self._closed = False
        self._finalized = False

        self.root.mkdir(parents=True, exist_ok=True)
        self._init_dataset_if_resuming()
        atexit.register(self._atexit_finalize)

    @property
    def num_episodes(self) -> int:
        return 0 if self._dataset is None else int(self._dataset.num_episodes)

    def _init_dataset_if_resuming(self) -> None:
        if not self.resume:
            return

        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            return

        LOGGER.info("Resuming existing LeRobot dataset at %s", self.root)
        self._dataset = LeRobotDataset(
            self.repo_id,
            root=self.root,
            batch_encoding_size=self.batch_encoding_size,
            vcodec=self.vcodec,
        )
        if self.image_writer_processes or self.image_writer_threads:
            self._dataset.start_image_writer(
                num_processes=self.image_writer_processes,
                num_threads=self.image_writer_threads,
            )

    def reset_buffer(self) -> None:
        self.buffer = []

    def add_observation(self, obs: Dict[str, Any]) -> None:
        obs_copy: Dict[str, Any] = {
            "joint_positions": np.asarray(obs["joint_positions"], dtype=np.float32),
            "next_joint": np.asarray(obs.get("next_joint", obs["joint_positions"]), dtype=np.float32),
        }

        # Keep camera handling permissive: use any available camera keys.
        for key in ("left_camera_rgb", "right_camera_rgb", "front_camera_rgb"):
            if key in obs:
                obs_copy[key] = np.asarray(obs[key], dtype=np.uint8)

        self.buffer.append(obs_copy)

    def _build_features_from_first_obs(self, first_obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        features: Dict[str, Dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": STATE_DIM_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": STATE_DIM_NAMES,
            },
        }

        for obs_key, cam_name in (
            ("left_camera_rgb", "left"),
            ("right_camera_rgb", "right"),
            ("front_camera_rgb", "front"),
        ):
            if obs_key not in first_obs:
                continue
            image = np.asarray(first_obs[obs_key])
            if image.ndim != 3:
                raise ValueError(f"Expected HxWxC image for {obs_key}, got shape {image.shape}")
            h, w, c = image.shape
            features[f"observation.images.camera_{cam_name}"] = {
                "dtype": "video",
                "shape": (int(h), int(w), int(c)),
                "names": ["height", "width", "channels"],
            }

        return features

    def _init_dataset_if_needed(self) -> None:
        if self._dataset is not None:
            return
        if not self.buffer:
            raise RuntimeError("Cannot initialize dataset from an empty episode buffer.")

        features = self._build_features_from_first_obs(self.buffer[0])
        if not any(key.startswith("observation.images.") for key in features):
            raise RuntimeError("No camera images found in observations.")

        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            robot_type=self.robot_type,
            features=features,
            use_videos=True,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
            batch_encoding_size=self.batch_encoding_size,
            vcodec=self.vcodec,
        )
        LOGGER.info("Created LeRobot dataset at %s", self.root)

    def save_episode(self, episode_data: Optional[List[Dict[str, Any]]] = None, task: Optional[str] = None) -> None:
        if self._closed:
            raise RuntimeError("Saver already closed.")

        data = episode_data if episode_data is not None else self.buffer
        if not data:
            LOGGER.warning("Empty episode, skipping save.")
            return

        self._init_dataset_if_needed()
        assert self._dataset is not None

        task_text = (task or self.language_instruction).strip() or "perform the task"
        for obs in data:
            frame: Dict[str, Any] = {
                "observation.state": np.asarray(obs["joint_positions"], dtype=np.float32),
                "action": np.asarray(obs.get("next_joint", obs["joint_positions"]), dtype=np.float32),
                "task": task_text,
            }
            if "left_camera_rgb" in obs:
                frame["observation.images.camera_left"] = obs["left_camera_rgb"]
            if "right_camera_rgb" in obs:
                frame["observation.images.camera_right"] = obs["right_camera_rgb"]
            if "front_camera_rgb" in obs:
                frame["observation.images.camera_front"] = obs["front_camera_rgb"]

            self._dataset.add_frame(frame)

        self._dataset.save_episode()
        LOGGER.info("Saved episode %s with %s frames.", self._dataset.num_episodes - 1, len(data))

        if episode_data is None:
            self.reset_buffer()

    def close(
        self,
        push_to_hub: bool = False,
        private: bool = False,
        tags: Optional[List[str]] = None,
    ) -> None:
        if self._closed:
            return

        if self._dataset is not None and not self._finalized:
            self._dataset.finalize()
            self._finalized = True
            LOGGER.info("Finalized LeRobot dataset at %s", self.root)

        if push_to_hub:
            if self._dataset is None:
                raise RuntimeError("No dataset exists to push.")
            self._dataset.push_to_hub(private=private, tags=tags)
            LOGGER.info("Pushed dataset to HF Hub: %s", self.repo_id)

        self._closed = True

    def _atexit_finalize(self) -> None:
        """
        Best-effort crash safety for graceful exits.
        Hard kills (SIGKILL/power loss) still lose in-flight episode data.
        """
        try:
            if not self._closed:
                self.close(push_to_hub=False)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Atexit finalize failed: %s", exc)
