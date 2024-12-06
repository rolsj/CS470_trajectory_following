"""
Script learns an agent for trajectory following. Policy strongly inspired by [1] 


[1] Penicka, Robert, et al. [*Learning minimum-time flight in cluttered environments.*](https://rpg.ifi.uzh.ch/docs/RAL_IROS22_Penicka.pdf) IEEE Robotics and Automation Letters 7.3 (2022): 7209-7216.
"""

import argparse
import numpy as np
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import TrajectoryFactory
from aviaries.configuration import Configuration
from aviaries.factories.uzh_trajectory_follower_factory import (
    TrajectoryFollowerAviaryFactory,
)

from agents.test_policy import run_test
from agents.train_policy import run_train
from runnables.utils.gen_eval_tracks import load_eval_tracks
from typing import Dict
from tqdm import tqdm
import json
from runnables.utils.utils import compute_metrics_single

from agents.regression import Regression
import os

###### INFRASTRUCTURE PARAMS #######
GUI = True
RECORD_VIDEO = False
OUTPUT_FOLDER = "checkpointed_models"
COLAB = False
DEFAULT_EVAL_SET_FOLDER = "./test_tracks/eval-v0_n-ctrl-points-3_n-tracks-20_2024-02-11_22:18:28_46929077-0248-4c6e-b2b1-da2afb13b2e2"
####################################

###### USUALLY NOT CHANGED #########
OBS = ObservationType("kin")  # 'kin' or 'rgb'
ACT = ActionType.ATTITUDE_PID
AGENTS = 1
NUM_DRONES = 1
CTRL_FREQ = 10
MA = False
####################################

###### TEST TRAIN FLAGS ############
TRAIN = True
VIS = True
TEST = True
####################################

###### ENVIRONMENT PARAMS ##########
TIMESTEPS = 2.5e6
N_ENVS = 20
EPISODE_LEN_SEC = 20
####################################

###### HYPERPARAMS #################
WAYPOINT_BUFFER_SIZE = 2
K_P = 5
K_WP = 8
K_S = 0.05
MAX_REWARD_DISTANCE = 0.0
WAYPOINT_DIST_TOL = 0.05
DEFAULT_DISCR_LEVEL = 10


####################################
def save_benchmark(benchmarks: Dict[str, float], file_path: str):
    with open(file_path, "w") as file:
        json.dump(benchmarks, file)

def generate_parabolic_trajectory(x_point, z_start, x_land, num_points, up):
    """
    Generate waypoints for a parabolic trajectory.
    
    Parameters:
        z_start (float): Starting height (a > 0)
        x_land (float): x-coordinate of the landing point
        num_points (int): Number of waypoints in the trajectory
        
    Returns:
        waypoints (list of tuples): List of (x, y, z) waypoints
    """
    a = -1.5
    c = z_start
    b = -(c/x_land)-a*(x_land)

    #x_shift = x_land/5
    # Coefficients for the parabola
    """
    a = -0.5  # Ensures z(x_land) = 0
    b = x_shift               # Symmetric parabola
    c = z_start - a*(b)**2                # Starting height
    """
    # Generate x values
    x_values = np.linspace(0, x_land, num_points)
    
    # Calculate z values for the trajectory
    z_values = a * (x_values)**2 + b*(x_values) + c
    
    # Generate waypoints
    if up:
        waypoints = [(x_point-x, 0, z) for x, z in zip(x_values, z_values)]
    else:
        waypoints = [(x_point+x, 0, z) for x, z in zip(x_values, z_values)]
    
    return waypoints

def init_targets(x_point, z_start, x_land, num_points, up):
    pts = generate_parabolic_trajectory(x_point,z_start,x_land,num_points,up)
    if up:
        pts = pts[::-1]
    initial_xyzs = np.array([pts[0]])
    t_wps = TrajectoryFactory.waypoints_from_numpy(pts)
    t_traj = TrajectoryFactory.get_discr_from_wps(t_wps)
    return t_traj, initial_xyzs

def determine_strategy(h1, h2, l) -> tuple[int, float]:
    data_dir = "./agents/train_data"
    with open(os.path.join(data_dir, "flight.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        x_train = np.array(data[0])
        y_train = np.array(data[1])
    model_flight = Regression()
    model_flight.train_with(x_train, y_train)
    
    with open(os.path.join(data_dir, "drive.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        x_train = np.array(data[0])
        y_train = np.array(data[1])
    model_drive = Regression()    
    model_drive.train_with(x_train, y_train)

    expected_cost_flight = model_flight.predict(np.array([[h1, h2, l]]))
    expected_cost_drive = model_drive.predict(np.array([[h1, h2, l]]))
    print(f"Expected cost (energy) when flight: {expected_cost_flight}")
    print(f"Expected cost (energy) when drive: {expected_cost_drive}")

    if expected_cost_flight < expected_cost_drive:
        print("flight selected")
        return (0, expected_cost_flight)
    else:
        print("drive selected")
        return (1, expected_cost_drive)

def run(
    output_folder=OUTPUT_FOLDER,
    gui=GUI,
    timesteps=TIMESTEPS,
    train: bool = TRAIN,
    test: bool = TEST,
    vis: bool = VIS,
    n_envs: int = N_ENVS,
    episode_len_sec: int = EPISODE_LEN_SEC,
    waypoint_buffer_size: int = WAYPOINT_BUFFER_SIZE,
    k_p: float = K_P,
    k_wp: float = K_WP,
    k_s: float = K_S,
    max_reward_distance: float = MAX_REWARD_DISTANCE,
    waypoint_dist_tol: float = WAYPOINT_DIST_TOL,
    discr_level: float = DEFAULT_DISCR_LEVEL,
    eval_set: set = DEFAULT_EVAL_SET_FOLDER,
):

    output_folder = f"{output_folder}/wp_b={waypoint_buffer_size}_k_p={k_p}_k_wp={k_wp}_k_s={k_s}_max_reward_distance={max_reward_distance}_waypoint_dist_tol={waypoint_dist_tol}"
    print(f"Output folder: {output_folder}")

    ##### Use regression models to determine the strategy #####
    selected_idx, expected_cost = determine_strategy(h1=5, h2=3, l=10)
    assert 0 <= selected_idx < 2

    ##### Set waypoints depending on the selected strategy #####
    if selected_idx == 0: # Flight mode
        raise Exception("FLIGHT mode trajectory not yet made")
    else: # Drive mode
        t_traj, init_wp = init_targets(0,1,1,5,False)
        t_traj1, init_wp1 = init_targets(2,1,1,5,True)

    config = Configuration(
        action_type=ACT,
        initial_xyzs=init_wp,
        output_path_location=output_folder,
        n_timesteps=timesteps,
        t_traj=t_traj,
        local=True,
        episode_len_sec=episode_len_sec,
        waypoint_buffer_size=waypoint_buffer_size,
        k_p=k_p,
        k_wp=k_wp,
        k_s=k_s,
        max_reward_distance=max_reward_distance,
        waypoint_dist_tol=waypoint_dist_tol,
    )

    env_factory = TrajectoryFollowerAviaryFactory(
        config=config,
        observation_type=OBS,
        use_gui_for_test_env=gui,
        n_env_training=n_envs,
        seed=0,
    )

    if train:
        run_train(config=config, env_factory=env_factory)

    if vis:
        run_test(
            k_p,
            k_wp,
            k_s,
            max_reward_distance,
            waypoint_dist_tol,
            t_traj1, config=config, env_factory=env_factory)

    if test:
        env_factory.single_traj = True
        env_factory.eval_mode = True
        tracks = load_eval_tracks(eval_set, discr_level=discr_level)
        all_visited_positions = []
        mean_devs = []
        max_devs = []
        successes = []
        times = []
        for track in tqdm(tracks):
            t_traj, init_wp = track, np.array([track[0].coordinate])
            config.update_trajectory(t_traj, init_wp)
            env_factory.set_config(config)
            visited_positions, success, time = run_test(
                k_p,
                k_wp,
                k_s,
                max_reward_distance,
                waypoint_dist_tol,
                t_traj1, config=config, env_factory=env_factory, eval_mode=True
            )
            successes.append(success)
            if success:
                mean_dev, max_dev = compute_metrics_single(visited_positions, track)
                mean_devs.append(mean_dev)
                max_devs.append(max_dev)
                all_visited_positions.append(visited_positions)
                times.append(time)
        print("SUCCESS RATE: ", np.mean(np.array(successes)))
        print("AVERAGE MEAN DEVIATION: ", np.mean(mean_devs))
        print("AVERAGE MAX DEVIATION: ", np.mean(max_devs))
        print("AVERAGE TIME UNTIL LANDING: ", np.mean(times))

        save_benchmark(
            {
                "success_rate": np.mean(successes),
                "avg mean dev": np.mean(mean_devs),
                "avg max dev": np.mean(max_devs),
                "avt time": np.mean(times),
            },
            f"rl_{discr_level}.json",
        )


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script"
    )
    parser.add_argument(
        "--gui",
        default=GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--timesteps",
        default=TIMESTEPS,
        type=int,
        help="number of train timesteps before stopping",
        metavar="",
    )
    parser.add_argument(
        "--train",
        default=TRAIN,
        type=str2bool,
        help="Whether to train (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--test",
        default=TEST,
        type=str2bool,
        help="Whether to test (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--vis",
        default=VIS,
        type=str2bool,
        help="Whether to visualise learned policy (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--n_envs",
        default=N_ENVS,
        type=int,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--episode_len_sec",
        default=EPISODE_LEN_SEC,
        type=int,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--waypoint_buffer_size",
        default=WAYPOINT_BUFFER_SIZE,
        type=int,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--k_p",
        default=K_P,
        type=float,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--k_wp",
        default=K_WP,
        type=float,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--k_s",
        default=K_S,
        type=float,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--max_reward_distance",
        default=MAX_REWARD_DISTANCE,
        type=float,
        help="number of parallel environments",
        metavar="",
    )
    parser.add_argument(
        "--waypoint_dist_tol",
        default=WAYPOINT_DIST_TOL,
        type=float,
        help="number of parallel environments",
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
