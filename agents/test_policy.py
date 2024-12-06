"""
Methods to test a policy.
"""

from aviaries.factories.base_factory import BaseFactory
from aviaries.configuration import Configuration
import os
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from aviaries.UZHAviary import UZHAviary
from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3 import PPO
import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync
import pybullet as p
from aviaries.rewards.uzh_trajectory_reward import Rewards

def test_simple_follower(
    k_p,
    k_wp,
    k_s,
    max_reward_distance,
    waypoint_dist_tol,
    trajectories,
    local: bool,
    filename: str,
    test_env: UZHAviary,
    output_folder: str,
    eval_mode=False,
):
    if os.path.isfile(filename + "/best_model.zip"):
        path = filename + "/best_model.zip"
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=False,
    )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    # 모든 궤적을 하나로 합치기
    all_waypoints = []
    for trajectory in trajectories:
        all_waypoints.extend(trajectory.wps)
    combined_trajectory = np.array([x.coordinate for x in all_waypoints])
    test_env.combined_trajectory = np.vstack(
            [
                combined_trajectory,
                np.array(combined_trajectory[-1] * np.ones((test_env.WAYPOINT_BUFFER_SIZE, 3))),
            ]
        )
    
    # 각 궤적별로 실행
    for trajectory in trajectories:
        test_env.current_waypoint_idx = 0
        test_env.single_traj = trajectory
        test_env.trajectory = test_env.set_trajectory()
        test_env.self_trajectory = test_env.set_trajectory()

        test_env.rewards = Rewards(
            trajectory=test_env.trajectory,
            k_p=k_p,
            k_wp=k_wp,
            k_s=k_s,
            max_reward_distance=max_reward_distance,
            dist_tol=waypoint_dist_tol,
        )
        test_env.rewards.reset(test_env.self_trajectory)

        test_env.furthest_reached_waypoint_idx = 0
        test_env.future_waypoints_relative = (
            test_env.trajectory[
                test_env.current_waypoint_idx : test_env.current_waypoint_idx
                + test_env.WAYPOINT_BUFFER_SIZE
            ]
            - test_env.trajectory[test_env.current_waypoint_idx]
        )

        for i in range((test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ):
            current_position = test_env._getDroneStateVector(0)[:3]
            target_position = test_env.trajectory[-1]
            distance_to_target = np.linalg.norm(current_position - target_position)

            if distance_to_target < 0.1:
                action = np.zeros_like(action)
                while not np.allclose(current_position[2], target_position[2], atol=0.015):
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    current_position = test_env._getDroneStateVector(0)[:3]
                print("Current trajectory completed")
                break
            else:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                obs2 = obs.squeeze()
                act2 = action.squeeze()

            logger.log(
                drone=0,
                timestamp=i / test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                control=np.zeros(12),
            )

            if not eval_mode:
                sync(i, start, test_env.CTRL_TIMESTEP)
            if terminated:
                if eval_mode:
                    test_env.close()
                    test_env.pos_logger.flush()
                    all_pos = test_env.pos_logger.load_all()
                    t = test_env.step_counter * test_env.PYB_TIMESTEP
                    success = test_env.reached_last_point if type(test_env) == UZHAviary else None
                    return all_pos, success, t
                obs = test_env.reset(seed=42, options={})
                break

    time.sleep(10)
    test_env.close()
    logger.plot()
    return None, False, None

def run_test(
    k_p,
    k_wp,
    k_s,
    max_reward_distance,
    waypoint_dist_tol,
    trajectories, config: Configuration, env_factory: BaseFactory, eval_mode=False):
    if not eval_mode:
        test_env = env_factory.get_test_env_gui()
    else:
        test_env = env_factory.get_test_env_no_gui()

    eval_res = test_simple_follower(
        k_p,
        k_wp,
        k_s,
        max_reward_distance,
        waypoint_dist_tol,
        trajectories,
        local=config.local,
        filename=config.output_path_location,
        test_env=test_env,
        output_folder=config.output_path_location,
        eval_mode=eval_mode,
    )
    return eval_res
