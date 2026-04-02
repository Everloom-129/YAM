#!/usr/bin/env python3
"""
Reconstruct one LeRobot episode into MolmoAct format (JSON + images).

Given a Hugging Face LeRobot dataset repo and an episode index, this script:
1) Loads split `train`.
2) Selects rows for `episode_index == <episode>`.
3) Exports all observation images for that episode.
4) Writes a MolmoAct-style JSON file with joint/action fields and image paths.

Output layout:
    /home/sean/Desktop/YAM/gello_software/data_recurrent/<dataset_name>/<episode_id>/
        <episode_id>.json
        <camera_name>_rgb/000000.png
        <camera_name>_rgb/000001.png
        ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from torchvision.io import VideoReader


DEFAULT_OUTPUT_ROOT = Path("/home/sean/Desktop/YAM/gello_software/data_recurrent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct one episode from a Hugging Face LeRobot dataset into MolmoAct format."
        )
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face dataset repo id, e.g. user/my_dataset",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index (integer).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional HF token for private repos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG and JSON outputs for this episode.",
    )
    return parser.parse_args()


def episode_strings(episode: int) -> tuple[str, str]:
    return str(episode), f"{episode:06d}"


def pick_output_dir(output_root: Path, repo_id: str, episode: int) -> Path:
    dataset_name = repo_id.split("/")[-1]
    ep6 = f"{episode:06d}"
    output_dir = output_root / dataset_name / ep6
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def split_dual_arm(vec: list[float]) -> tuple[list[float], list[float]]:
    if len(vec) >= 14:
        return vec[:7], vec[7:14]
    half = len(vec) // 2
    return vec[:half], vec[half:]


def _to_float_list(value: Any) -> list[float]:
    # Handles lists, numpy arrays, torch tensors via .tolist when available.
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"Expected list-like value, got {type(value)}")
    return [float(v) for v in value]


def camera_suffix_from_video_key(video_key: str) -> str:
    # video_key looks like "observation.images.camera_front"
    suffix = video_key.split("observation.images.", 1)[1]
    if suffix.startswith("camera_"):
        suffix = suffix[len("camera_") :]
    return suffix


def _download(repo_id: str, filename: str, revision: str | None, token: str | None) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        token=token,
    )


def load_repo_metadata(
    repo_id: str, revision: str | None, token: str | None
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    info_path = _download(repo_id, "meta/info.json", revision, token)
    episodes_path = _download(repo_id, "meta/episodes/chunk-000/file-000.parquet", revision, token)

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    episodes_df = pd.read_parquet(episodes_path)

    # Prefer files referenced in episodes metadata to avoid many failing downloads.
    used_pairs = sorted(
        {
            (int(row["data/chunk_index"]), int(row["data/file_index"]))
            for _, row in episodes_df.iterrows()
        }
    )
    candidate_files = [f"data/chunk-{c:03d}/file-{f:03d}.parquet" for c, f in used_pairs]

    data_frames: list[pd.DataFrame] = []
    for filename in candidate_files:
        local = _download(repo_id, filename, revision, token)
        data_frames.append(pd.read_parquet(local))
    if not data_frames:
        raise ValueError("No data parquet files found in dataset.")
    data_df = pd.concat(data_frames, ignore_index=True)
    return info, episodes_df, data_df


def reconstruct_episode_from_split(
    repo_id: str,
    episode: int,
    output_dir: Path,
    revision: str | None,
    token: str | None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    info, episodes_df, data_df = load_repo_metadata(repo_id, revision, token)
    ep_rows = episodes_df[episodes_df["episode_index"] == int(episode)]
    if ep_rows.empty:
        available = sorted(int(x) for x in episodes_df["episode_index"].tolist())
        raise ValueError(f"Episode {episode} not found. Available episodes: {available[:20]}")

    ep_meta = ep_rows.iloc[0]
    ep_data = data_df[data_df["episode_index"] == int(episode)].sort_values("frame_index")
    if ep_data.empty:
        raise ValueError(f"No frame rows found in data parquet for episode {episode}.")

    fps = int(info.get("fps", 30))
    features = info.get("features", {})
    video_keys = sorted([k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"])
    if not video_keys:
        raise ValueError("No video features found in dataset metadata.")

    camera_dirs: dict[str, Path] = {}
    for vkey in video_keys:
        cam = camera_suffix_from_video_key(vkey)
        cam_dir = output_dir / f"{cam}_rgb"
        cam_dir.mkdir(parents=True, exist_ok=True)
        camera_dirs[vkey] = cam_dir

    # Pre-extract all camera frames for the episode.
    frame_paths: dict[str, list[str]] = {}
    episode_len = int(ep_meta["length"])
    for vkey in video_keys:
        cam = camera_suffix_from_video_key(vkey)
        chunk_idx = int(ep_meta[f"videos/{vkey}/chunk_index"])
        file_idx = int(ep_meta[f"videos/{vkey}/file_index"])
        from_ts = float(ep_meta[f"videos/{vkey}/from_timestamp"])
        to_ts = float(ep_meta[f"videos/{vkey}/to_timestamp"])
        video_rel = f"videos/{vkey}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
        video_path = _download(repo_id, video_rel, revision, token)

        reader = VideoReader(video_path, "video")
        reader.seek(from_ts)
        saved_paths: list[str] = []
        local_idx = 0
        for packet in reader:
            pts = float(packet["pts"])
            if pts + (0.5 / max(1, fps)) < from_ts:
                continue
            if pts > to_ts + (0.5 / max(1, fps)):
                break

            frame_rgb = packet["data"].permute(1, 2, 0).numpy()  # HWC RGB uint8
            img_path = camera_dirs[vkey] / f"{local_idx:06d}.png"
            cv2.imwrite(
                str(img_path),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            saved_paths.append(str(img_path))
            local_idx += 1
            if local_idx >= episode_len:
                break

        if len(saved_paths) != episode_len:
            raise RuntimeError(
                f"Decoded {len(saved_paths)} frames for {video_rel}, expected {episode_len}."
            )
        frame_paths[cam] = saved_paths

    records: list[dict[str, Any]] = []
    for frame_idx, (_, row) in enumerate(ep_data.iterrows()):
        state = _to_float_list(row["observation.state"].tolist() if isinstance(row["observation.state"], np.ndarray) else row["observation.state"])
        action = _to_float_list(row["action"].tolist() if isinstance(row["action"], np.ndarray) else row["action"])
        left_joint, right_joint = split_dual_arm(state)
        next_left_joint, next_right_joint = split_dual_arm(action)

        tasks = ep_meta.get("tasks", [])
        if hasattr(tasks, "tolist"):
            tasks = tasks.tolist()
        task = tasks[0] if isinstance(tasks, list) and tasks else ""
        task = task if isinstance(task, str) else str(task)

        record: dict[str, Any] = {
            "language_instruction": task,
            "left_joint": json.dumps(left_joint),
            "right_joint": json.dumps(right_joint),
            "next_left_joint": json.dumps(next_left_joint),
            "next_right_joint": json.dumps(next_right_joint),
        }

        for cam, paths in frame_paths.items():
            record[f"image_{cam}_rgb"] = paths[frame_idx]

        records.append(record)
    return records, sorted(frame_paths.keys()), episode_len


def main() -> None:
    args = parse_args()
    output_dir = pick_output_dir(args.output_root, args.repo_id, args.episode)
    _, ep6 = episode_strings(args.episode)
    output_path = output_dir / f"{ep6}.json"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output JSON already exists: {output_path}. Use --overwrite to regenerate."
        )

    data, cameras, episode_len = reconstruct_episode_from_split(
        repo_id=args.repo_id,
        episode=args.episode,
        output_dir=output_dir,
        revision=args.revision,
        token=args.token,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("Reconstructed episode from LeRobot parquet/video into MolmoAct format.")
    print(f"Saved episode {args.episode} JSON to: {output_path}")
    print(f"Saved cameras: {', '.join(cameras) if cameras else '(none found)'}")
    print(f"Saved {episode_len} frames per camera.")


if __name__ == "__main__":
    main()
