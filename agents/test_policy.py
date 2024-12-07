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
import xml.etree.ElementTree as ET

def compute_engery(env, action, dt, rpm_prev, k_m):
    '''
    ** params **
    env: env - test environment
    action: action - test action
    dt: float - time period
    rpm_prev: ndarray - 1 * 4 size rpm
    km: float - thrust coeff.

    ** return **
    energy: float - energy
    rpm_prev: ndarray - 1 * 4 size rpm
    '''
    # compute energy expenditure
    rpm = env.last_clipped_action
    ang_vel = rpm * np.pi / 30
    ang_acc = rpm - rpm_prev
    
    # energy term 1 ( sum{k_m * w^2} * dt )
    J1 = dt * np.sum(k_m * ang_vel ** 2)

    # energy term 2 ( sum{acc * k_m * (acc - k_m * w)} * dt )
    J2 = dt * np.sum(ang_acc * k_m * (ang_acc - k_m * ang_vel))

    # choose either J1 or J2 as energy
    return 1, rpm

def test_simple_follower(
    k_p,
    k_wp,
    k_s,
    h1,
    h2,
    l,
    max_reward_distance,
    waypoint_dist_tol,
    trajectories,
    local: bool,
    filename: str,
    test_env: UZHAviary,
    output_folder: str,
    eval_mode=False,
    map_name=None,
):
    if os.path.isfile(filename + "/best_model.zip"):
        path = filename + "/best_model.zip"
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)
    
    # load drone urdf model
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
    k_m = float(drone_properties["km"])
    time_period = 1 / test_env.CTRL_FREQ
    
    # load map urdf model
    if map_name:
        tree = ET.parse(f"../gym_pybullet_drones/assets/{map_name}.urdf")
        root = tree.getroot()
        drone_properties = {}
    
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
    
    # PID 제어 관련 상수 추가
    P_COEFF_WHEEL = 50
    I_COEFF_WHEEL = 0.1
    D_COEFF_WHEEL = 10
    last_x_error = 0
    integral_x_error = 0
    
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

        rpm_prev = np.zeros((1, 4))
        energy_cum = 0.
        energy_cur = 0.
        for i in range((test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ):
            current_position = test_env._getDroneStateVector(0)[:3]
            target_position = test_env.trajectory[-1]
            distance_to_target = np.linalg.norm(current_position - target_position)
            
            if distance_to_target < 0.13:
                action = np.zeros_like(action)
                while not np.allclose(current_position[2], target_position[2], atol=0.05):
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    current_position = test_env._getDroneStateVector(0)[:3]
                print("Current trajectory completed")
                break
            else:
                states = test_env._getDroneStateVector(0)
                action, _states = model.predict(obs, deterministic=True)
                
                # 현재 높이와 다음 웨이포인트 높이 확인
                if current_position[0] < 0.05:
                    altitude = h1
                elif current_position[1] > l - 0.05:
                    altitude = h2
                else:
                    altitude = 0
                current_height = current_position[2] - altitude
                current_projection, current_projection_idx, reached_distance = test_env.rewards.get_travelled_distance(current_position)
                next_waypoint_idx = min(current_projection_idx + 1, len(test_env.trajectory) - 1)
                next_waypoint_height = test_env.trajectory[next_waypoint_idx][2] - altitude
                
                # 지상 이동 가능 여부 판단
                is_on_ground = current_height <= 0.2
                next_point_near_ground = next_waypoint_height <= 0.1
                can_use_ground = is_on_ground and next_point_near_ground
                
                if can_use_ground:
                    # PID 제어를 통한 바퀴 속도 계산
                    x_error = test_env.trajectory[next_waypoint_idx][0] - current_position[0]
                    integral_x_error += x_error * (1.0/test_env.CTRL_FREQ)
                    derivative_x_error = (x_error - last_x_error) * test_env.CTRL_FREQ
                    last_x_error = x_error
                    
                    base_velocity = P_COEFF_WHEEL * x_error + \
                                  I_COEFF_WHEEL * integral_x_error + \
                                  D_COEFF_WHEEL * derivative_x_error
                    
                    # 바퀴 속도 제한
                    base_velocity = np.clip(base_velocity, -10.0, 10.0)
                    wheel_velocities = np.array([base_velocity] * 4)
                    
                    # 바퀴 제어 적용
                    for j, wheel_id in enumerate(test_env.wheel_joints):
                        p.setJointMotorControl2(
                            test_env.DRONE_IDS[0],
                            wheel_id,
                            p.VELOCITY_CONTROL,
                        targetVelocity=wheel_velocities[j],
                    )
                    action = np.zeros(4)
                    action = action.reshape(1, 4)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    print("hello Using PID wheel control, velocity:", base_velocity)
                else:
                    # 공중에서는 일반 동작
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    if is_on_ground:
                        print("hello Taking off for next waypoint")
                
                obs2 = obs.squeeze()
                act2 = action.squeeze()
            
            #if i % test_env.CTRL_FREQ == 0:    
            energy_cur, rpm_prev = compute_engery(test_env, action, time_period, rpm_prev, k_m)
            energy_cum += energy_cur

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
    h1,
    h2,
    l,
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
        h1,
        h2,
        l,
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
