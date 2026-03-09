# Collect data for a single task. Pair with oculus viser agent.
import os
import sys
from tkinter import Y

sys.path.append('/home/sean/Desktop/YAM')
sys.path.append('/home/sean/Desktop/YAM/yam_realtime')
import numpy as np

from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.launch_utils import cleanup_processes, initialize_agent, initialize_robots, initialize_sensors, setup_can_interfaces, setup_logging
from yam_realtime.utils.camera_thread import EpisodeSaverThread

import tqdm
from yam_realtime.envs.robot_env import RobotEnv
from yam_realtime.utils.data_saver_joint import DataSaverJoint

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.robots.robot import Robot
import logging
import time
import tyro

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    collection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_record_replay.yaml",)
    

def main(args: Args):
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")

    server_processes = []
    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")
        logger.info(f"camera_dict: {camera_dict}")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=rate,
        )

        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')
        

        logger.info("Starting control loop...")
        data_saver = DataSaverJoint(
            save_dir=configs_dict["storage"]["base_dir"],
            task_directory=configs_dict["storage"]["task_directory"],
            language_instruction=configs_dict["storage"]["language_instruction"],
            saver_max_workers=configs_dict["storage"].get("saver_max_workers"),
            png_compress_level=configs_dict["storage"].get("png_compress_level", 1),
        )
        _run_control_loop(env, agent, main_config, configs_dict, data_saver)


    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, resetting robots before shutdown...")
        if "env" in locals() and "agent" in locals():
            try:
                reset_robot(agent, env, "left")
                reset_robot(agent, env, "right")
            except Exception as reset_error:
                logger.warning(f"Failed to reset robots during shutdown: {reset_error}")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if "env" in locals():
            env.close()
        if "agent" in locals():
            cleanup_processes(agent, server_processes)


# slowly move the robot back to original position
def reset_robot(agent: Agent, env: RobotEnv, side: str):
    agent.act({})
    current_pos = env.robot(side).get_joint_pos()
    target_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    steps = 50
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
        env.robot(side).command_joint_pos(target_pos)
        time.sleep(2 / steps)

def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig, configs_dict: Dict, data_saver: DataSaverJoint) -> None:
    """
    Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    previous_action = agent.act({})
    saver_thread = EpisodeSaverThread(data_saver)
    saver_thread.start()

    logger = logging.getLogger(__name__)
    num_traj = data_saver.traj_count

    # Init environment and warm up agent
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")
    # Main control loop
    while num_traj <= configs_dict['storage']['episodes']:
        obs = env.reset()
        data_saver.reset_buffer()
        

        logger.info(f"Press 'A' to start collecting data: ")

        while True:
            info = agent.get_info()
            if info["success"]:
                logger.info(f"Successfully pressed 'A', starting to collect data")

                time.sleep(2)
                break
        logger.info(f"Press 'A' to save the data, press 'B' to discard the data")

        for _ in tqdm.tqdm(range(configs_dict['collection']['max_episode_length']), desc=f"Collecting data {num_traj}/{configs_dict['storage']['episodes']}"):
            info = agent.get_info()
            while (not info["success"] and not info["failure"]) and not info["movement_enabled"]['left'] and not info["movement_enabled"]['right']:
                info = agent.get_info()
            
            save = False
            if info["success"]:
                save = True
                break
            elif info["failure"]:
                break

            # within agent, we use 7 point representation for action (x, y, z, w, x, y, z) for both left and right arm
            # but for robot to actually move, we use 6 value joints position instead. So the act function in agent converts 
            # the 7 points action. Also, there are some nasty conversion for quat from (w, x, y, z) to (x, y, z, w),
            # the robot and viser uses (w, x, y, z) for quat. but python scipy uses (x, y, z, w) for quat.

            # currently in the json for traning, the delta acton is actually a 7 point action without gripper. (from calc_delta_action)
            # Also make sure that gripper is either 0 or 1. 


            # note that we didn't really use obs (this is the outside world joints state containing pos, gripper, eff, vel)
            # we use viser agent to compute the action in 3D viser space, which is then solved by ik to get the joints position.
            # The action that the env.step() uses is the viser joints (the action returned by agent.act()).
            if info["movement_enabled"]['left'] or info["movement_enabled"]['right']:
                act = agent.act(obs)
                action = {'left': {'pos':act['left']['pos']}, 'right': {'pos':act['right']['pos']}}
                next_obs = env.step(action)
                data_saver.add_observation(obs, next_obs)
                obs = next_obs.copy()
        
        if save:
            if data_saver.buffer == []:
                logger.info(f"No data collected, skipping save")
                continue
            # data_saver.save_episode_json()
            saver_thread.save_episode(data_saver.buffer.copy())
            num_traj +=1
            logger.info(f"Successfully collected data")
        else:
            logger.info(f"Failure")
        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')

    saver_thread.stop()
    saver_thread.join()

    env.reset()
    logger.info(f"Finished collecting data")

if __name__ == "__main__":
    main(tyro.cli(Args))
