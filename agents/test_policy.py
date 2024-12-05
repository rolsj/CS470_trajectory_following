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


def test_simple_follower(
    local: bool,
    filename: str,
    test_env: BaseRLAviary,
    output_folder: str,
    eval_mode=False,
):

    # load model
    if os.path.isfile(filename + "/best_model.zip"):
        path = filename + "/best_model.zip"
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    # visualise in test environment
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=False,
    )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ):
        # 드론의 현재 위치와 목표 위치 계산
        current_position = test_env._getDroneStateVector(0)[:3]
        target_position = test_env.trajectory[-1]
        distance_to_target = np.linalg.norm(current_position - target_position)
        if distance_to_target < 0.1:
            action = np.zeros_like(action)
            while (np.allclose(current_position[2], target_position[2], atol=0.015)) == False:
                obs, reward, terminated, truncated, info = test_env.step(action)
                current_position = test_env._getDroneStateVector(0)[:3]
            print("done")
            break
        else:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            obs2 = obs.squeeze()
            act2 = action.squeeze()

        # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
            control=np.zeros(12),
        )

        # test_env.render()
        # print(terminated)
        if not eval_mode:
            sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            print("terminated")
            if eval_mode:
                test_env.close()
                test_env.pos_logger.flush()
                all_pos = test_env.pos_logger.load_all()
                t = test_env.step_counter * test_env.PYB_TIMESTEP
                if type(test_env) == UZHAviary:
                    success = test_env.reached_last_point
                else:
                    success = None
                return all_pos, success, t
            obs = test_env.reset(seed=42, options={})
            break
    """
    test_env.single_traj = t_traj3
    test_env.set_trajectory()
    
    for i in range((test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ):
        # 드론의 현재 위치와 목표 위치 계산
        current_position = test_env._getDroneStateVector(0)[:3]
        target_position = test_env.trajectory[-1]
        distance_to_target = np.linalg.norm(current_position - target_position)
        if distance_to_target < 0.1:
            action = np.zeros_like(action)
            while (np.allclose(current_position[2], target_position[2], atol=0.015)) == False:
                obs, reward, terminated, truncated, info = test_env.step(action)
                current_position = test_env._getDroneStateVector(0)[:3]
            print("done")
            break
        else:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            obs2 = obs.squeeze()
            act2 = action.squeeze()

        # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
            control=np.zeros(12),
        )

        # test_env.render()
        # print(terminated)
        if not eval_mode:
            sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            print("terminated")
            if eval_mode:
                test_env.close()
                test_env.pos_logger.flush()
                all_pos = test_env.pos_logger.load_all()
                t = test_env.step_counter * test_env.PYB_TIMESTEP
                if type(test_env) == UZHAviary:
                    success = test_env.reached_last_point
                else:
                    success = None
                return all_pos, success, t
            obs = test_env.reset(seed=42, options={})
            break
    """
    test_env.close()
    logger.plot()
    return None, False, None

def run_test(config: Configuration, env_factory: BaseFactory, eval_mode=False):
    if not eval_mode:
        test_env = env_factory.get_test_env_gui()
    else:
        test_env = env_factory.get_test_env_no_gui()

    eval_res = test_simple_follower(
        local=config.local,
        filename=config.output_path_location,
        test_env=test_env,
        output_folder=config.output_path_location,
        eval_mode=eval_mode,
    )
    return eval_res
