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
    time_period = 1 / test_env.CTRL_FREQ
    rpm_prev = np.zeros((test_env.NUM_DRONES, 4))
    acc_cum = 0.
    for i in range((test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        # 드론의 현재 위치와 목표 위치 계산
        current_position = test_env._getDroneStateVector(0)[:3]
        target_position = test_env.trajectory[-1]
        distance_to_target = np.linalg.norm(current_position - target_position)

        rpm = np.reshape(test_env._preprocessAction(action), (test_env.NUM_DRONES, 4))
        rpm_acc = rpm - rpm_prev
        rpm_prev = rpm
        acc_cum += time_period * rpm_acc / 60.

        # 자유 낙하 조건
        if distance_to_target < 0.1:  # 목적지와 0.5m 이내로 접근하면
            print("Entering free fall...")
            current_z = test_env._getDroneStateVector(0)[2]
            print(current_z)
            while current_z > 0.0:  # 자유 낙하 루프
                action = np.zeros_like(action)  # 모든 제어 신호를 0으로 설정
                obs, reward, terminated, truncated, info = test_env.step(action)

                # 드론의 현재 z좌표 확인
                current_z = test_env._getDroneStateVector(0)[2]  # z좌표 (x=0, y=1, z=2)

                # z좌표가 0 이하로 떨어지면 자유 낙하 종료

                # 중간 지연을 추가해 시뮬레이션 속도 조절
            time.sleep(10)
            break  # 루프 종료 (필요 시)

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
    test_env.close()
    print("here1")
    print("[total rpm_acc]", rpm_acc)
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
