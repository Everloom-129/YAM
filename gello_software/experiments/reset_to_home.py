"""Reset the YAM arm(s) to their home (start) joint position.

Builds the robot(s) straight from the launch config(s) — no cameras, no GELLO
leader, no policy — and smoothly interpolates to ``agent.start_joints`` using
the same ``move_to_start_position`` helper the launchers use. Handy after an
eval/teleop run leaves the arm somewhere awkward, or any time you want a known
pose before powering down.

The motors must be live first: run the CAN reset + ``set_timeout.py`` startup
sequence (see CLAUDE.md) so the arms hold position.

CLI::

    # bimanual (default): both arms to home
    python -m experiments.reset_to_home

    # single arm
    python -m experiments.reset_to_home --left-only

    # explicit configs
    python -m experiments.reset_to_home \
        --left_config_path configs/yam_left.yaml \
        --right_config_path configs/yam_right.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tyro
from omegaconf import OmegaConf

from gello.env import RobotEnv
from gello.robots.robot import BimanualRobot
from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position


@dataclass
class Args:
    left_config_path: str = "configs/yam_left.yaml"
    """Path to the left arm configuration YAML file."""

    right_config_path: str = "configs/yam_right.yaml"
    """Path to the right arm configuration YAML file (used unless --left-only)."""

    left_only: bool = False
    """Reset only the left arm (single-arm setups)."""


def _build_robot_from_cfg(cfg: Dict[str, Any]):
    """Instantiate the robot from a config's ``robot`` block.

    Mirrors the launchers: if ``robot.config`` is a path string, expand it to a
    dict before instantiating.
    """
    robot_cfg = cfg["robot"]
    if isinstance(robot_cfg.get("config"), str):
        robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(robot_cfg["config"]), resolve=True
        )
    return instantiate_from_dict(robot_cfg)


def build_home_env(
    args: Args,
) -> Tuple[RobotEnv, Dict[str, Any], Optional[Dict[str, Any]], bool]:
    """Build a camera-less RobotEnv over one or both arms.

    Returns ``(env, left_cfg, right_cfg, bimanual)``.
    """
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.left_config_path), resolve=True)
    bimanual = not args.left_only

    right_cfg: Optional[Dict[str, Any]] = None
    left_robot = _build_robot_from_cfg(left_cfg)
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )
        right_robot = _build_robot_from_cfg(right_cfg)
        robot = BimanualRobot(left_robot, right_robot)
    else:
        robot = left_robot

    # No cameras: home-ing only needs joint state + position commands.
    env = RobotEnv(robot, control_rate_hz=left_cfg.get("hz", 30))
    return env, left_cfg, right_cfg, bimanual


def main(args: Args) -> None:
    env, left_cfg, right_cfg, bimanual = build_home_env(args)
    try:
        print(f"Resetting {'both arms' if bimanual else 'left arm'} to home position...")
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
        print("Done. Arm(s) at home position.")
    finally:
        robot = env.robot()
        if hasattr(robot, "close"):
            robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
