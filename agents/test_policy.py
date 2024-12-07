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

def compute_engery(env, dt, rpm, rpm_prev, k_m):
    '''
    ** params **
    env: env - test environment
    dt: float - time period
    rpm: ndarray - 1 * 4 size rpm
    rpm_prev: ndarray - 1 * 4 size rpm
    km: float - thrust coeff.

    ** return **
    energy: float - energy
    rpm_prev: ndarray - 1 * 4 size rpm
    '''
    # compute energy expenditure
    rpm = np.reshape(env._preprocessAction(action), (1, 4))
    ang_vel = rpm * np.pi() / 30
    ang_acc = rpm - rpm_prev
    
    # energy term 1 ( sum{k_m * w^2} * dt )
    J1 = dt * np.sum(k_m * ang_vel ** 2)

    # energy term 2 ( sum{acc * k_m * (acc - k_m * w)} * dt )
    J2 = dt * np.sum(ang_acc * k_m * (ang_acc - k_m * ang_vel))

    # choose either J1 or J2 as energy
    return J1, rpm
    
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

    # load urdf model
    tree = ET.parse("../gym_pybullet_drones/assets/cf2x.urdf")
    root = tree.getroot()
    
    drone_properties = {}
    for element in root.iter('properties'):
        # Extract attributes from the <properties> tag
        # arm                 [m] : distance from center of mass to each motor
        # kf              [N s^2] : thrust coeff.
        # km            [N m s^2] : torque coeff.
        # thrust2weight           : thrust / weight
        # max_speed_kmh    [km/h] : max speed
        # gnd_eff_coeff           : ground effect coeff.
        # prop_radius         [m] : radius of each motor
        # drag_coeff_xy           : drag on xy plane coeff.
        # drag_coeff_z            : drag on z axis coeff.
        # dw_coeff_1              : drag of wing coeff.1
        # dw_coeff_2              : drag of wing coeff.2
        # dw_coeff_3              : drag of wing coeff.3
        drone_properties.update(element.attrib)
    k_m = drone_properties["km"]

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=False,
    )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
<<<<<<< HEAD
    for i in range((test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ):
        # 드론의 현재 위치와 목표 위치 계산
        current_position = test_env._getDroneStateVector(0)[:3]
        target_position = test_env.trajectory[-1]
        distance_to_target = np.linalg.norm(current_position - target_position)
        print(test_env.future_waypoints_relative)
        if distance_to_target < 0.14:
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
        
        # cumulating energy
        energy, rpm_prev = compute_engery(test_env, time_period, rpm, rpm_prev, k_m)
        energy_cum += energy

        # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
            control=np.zeros(12),
=======
    
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
>>>>>>> ffb7809781c31bf1af3ddbae591e82d851c84e2b
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
<<<<<<< HEAD
            #print(test_env.rewards.get_travelled_distance(current_position))
            #print(test_env.current_projection_idx)
            print(test_env.future_waypoints_relative)
=======

>>>>>>> ffb7809781c31bf1af3ddbae591e82d851c84e2b
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
