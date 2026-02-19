#!/usr/bin/env python
"""
Convert MolmoAct-style dataset to LeRobot v3.0 format in one shot.

This follows the same high-level pipeline and logic as molmoact_to_lerobot_v21.py:
  1. Episode-first layout on disk:
        data_dir/
        ├── 000001/
        │   ├── 000001.json
        │   ├── left_rgb/
        │   ├── right_rgb/
        │   └── front_rgb/
        ├── 000002/
        │   ├── 000002.json
        │   ├── left_rgb/
        │   ├── right_rgb/
        │   └── front_rgb/
        └── ...

  2. Load all episodes into memory (qpos, actions, images).
  3. Stream frames into a LeRobotDataset via `add_frame` + `save_episode`.

Differences vs the v2.1 script:
  - Uses the v3.0 LeRobotDataset API from `lerobot.datasets.lerobot_dataset`.
  - Creates a v3.0 dataset directly (no v2.1→v3.0 conversion step).
  - Calls `dataset.finalize()` at the end to produce a valid v3.0 dataset.
  - Does NOT support resume into an existing dataset directory; the output_dir must
    be new or empty.

Usage example:

    python molmoact_to_lerobot_v30.py \
        --data_dir /path/to/molmoact \
        --output_dir /path/to/molmoact_lerobot_v30 \
        --repo_id your-user/molmoact_v30 \
        --fps 10

    hf upload williamtsai726/stop_the_rolling_glue_rlds_0203 ./stop_the_rolling_glue_rlds --repo-type=dataset

You can then train with:

    # diffusion policy
    python src/lerobot/scripts/lerobot_train.py \
            --dataset.repo_id=williamtsai726/stop_the_rolling_glue_0206 \
            --policy.type=diffusion \
            --policy.repo_id=williamtsai726/stop_the_rolling_glue_0206 \
            --output_dir=./outputs/stop_the_rolling_glue_0206 \
            --save_after_step=60000 \
            --steps=100000 \
            
            --resume=true \
            --config_path=/home/sean/Desktop/YAM/lerobot/outputs/stack_cube_into_pyramid_0203/checkpoints/100000/pretrained_model/train_config.json

Note: For local training, `--dataset.repo_id` can be the absolute path to the dataset directory.
"""

#!/usr/bin/env python
"""
Convert MolmoAct-style dataset to LeRobot v3.0 format.
Fixed to avoid OOM by using Lazy Loading of images.
"""

import argparse
import json
import os
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import tqdm
from tqdm import trange

# LeRobot v3.0 API
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_molmoact_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load MolmoAct episodes, but store image PATHS instead of pixel data
    to save memory during the initial load.
    """
    episodes: List[Dict[str, Any]] = []
    data_path = Path(data_dir)

    episode_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    print(f"Found {len(episode_dirs)} episodes under {data_dir}")

    for ep_dir in episode_dirs:
        episode_id = ep_dir.name
        json_path = ep_dir / f"{episode_id}.json"

        if not json_path.exists():
            continue

        with open(json_path, "r") as f:
            episode_data = json.load(f)

        if not episode_data:
            continue

        first_frame = episode_data[0]
        task_description = first_frame.get("task", f"task_{episode_id}")

        # Joint positions and Actions
        left_joint = np.array([json.loads(f["left_joint"]) for f in episode_data], dtype=np.float32)
        right_joint = np.array([json.loads(f["right_joint"]) for f in episode_data], dtype=np.float32)
        
        qpos = np.concatenate([left_joint, right_joint], axis=1)   # (T, 14)
        actions = qpos.copy() # Standard practice for MolmoAct to v2.1/v3.0

        episode_info: Dict[str, Any] = {
            "task_description": task_description,
            "qpos": qpos,
            "actions": actions,
            "episode_length": len(actions),
            "camera_paths": {}, # Changed from "images": [] to a path map
        }

        for camera_dir in [ep_dir / "left_rgb", ep_dir / "right_rgb", ep_dir / "front_rgb"]:
            if not camera_dir.exists():
                continue

            cam_name = camera_dir.name.replace("_rgb", "")
            image_files = sorted(
                [f for f in camera_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            )
            # Store only the paths to keep RAM usage near zero here
            episode_info["camera_paths"][cam_name] = image_files

        episodes.append(episode_info)

    print(f"Metadata loaded for {len(episodes)} episodes.")
    return episodes


def infer_camera_shapes(episodes: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int, int]]:
    """Inspect exactly one image to determine H, W, C for the schema."""
    for ep in episodes:
        for cam_name, paths in ep["camera_paths"].items():
            if paths:
                with Image.open(paths[0]) as img:
                    w, h = img.size
                    c = len(img.getbands())
                    return {"left": (h, w, c), "right": (h, w, c), "front": (h, w, c)}
    return {"left": (360, 640, 3), "right": (360, 640, 3), "front": (360, 640, 3)}


def create_lerobot_dataset_v30(episodes, output_dir, repo_id, fps, robot_type):
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"Output directory '{output_dir}' is not empty.")

    camera_shapes = infer_camera_shapes(episodes)
    image_dim_names = ["height", "width", "channels"]
    
    # Feature schema (same keys and shapes as the v2.1 script; v3.0 layout is handled by LeRobot).
    features: Dict[str, Dict[str, Any]] = {
        # Robot joint positions
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
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
            ],
        },
        # Actions (joint-space)
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_m1",
                "left_m2",
                "left_m3",
                "left_m4",
                "left_m5",
                "left_m6",
                "left_m7",
                "right_m8",
                "right_m9",
                "right_m3",
                "right_m4",
                "right_m5",
                "right_m6",
                "right_m7",
            ],
        },
        # Image observations
        "observation.images.camera_left": {
            "dtype": "image",
            "shape": camera_shapes["left"],
            "names": image_dim_names,
        },
        "observation.images.camera_right": {
            "dtype": "image",
            "shape": camera_shapes["right"],
            "names": image_dim_names,
        },
        "observation.images.camera_front": {
            "dtype": "image",
            "shape": camera_shapes["front"],
            "names": image_dim_names,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=True, 
    )

    for ep_idx, ep_data in enumerate(tqdm.tqdm(episodes, desc="Processing Episodes")):
        qpos = ep_data["qpos"]
        actions = ep_data["actions"]
        cam_paths = ep_data["camera_paths"]

        for f_idx in trange(ep_data["episode_length"], leave=False, desc=f"Frames (ep {ep_idx})"):
            if f_idx < 5: continue # Skip initial frames as per original script

            frame_data = {
                "observation.state": qpos[f_idx],
                "action": actions[f_idx],
                "task": ep_data["task_description"],
            }

            # JUST-IN-TIME IMAGE LOADING
            for cam_name, paths in cam_paths.items():
                if f_idx < len(paths):
                    # Open only this specific frame's image
                    with Image.open(paths[f_idx]) as img:
                        frame_data[f"observation.images.camera_{cam_name}"] = img.convert("RGB")

            dataset.add_frame(frame_data)

        # Finalize the episode on disk and clear temp buffers
        dataset.save_episode()
        
        # Explicitly trigger garbage collection after each episode
        gc.collect()

    print("Finalizing v3.0 dataset (writing Parquet and MP4 files)...")
    dataset.finalize()
    print(f"Success! Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="molmoact_v30")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot_type", type=str, default="molmoact_dual_arm")
    args = parser.parse_args()

    episodes = load_molmoact_data(args.data_dir)
    create_lerobot_dataset_v30(episodes, args.output_dir, args.repo_id, args.fps, args.robot_type)


if __name__ == "__main__":
    main()