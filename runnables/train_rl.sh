#!/bin/bash

python ../agents/C_trajectory_following.py \
    --gui "False" \
    --output_folder "./weights" \
    --timesteps 100000 \
    --train "True" \
    --test "False" \
    --vis "False" \
    --n_envs "5" \
    --episode_len_sec "20" \
    --waypoint_buffer_size "2" \
    --k_p "5.0" \
    --k_wp "8.0" \
    --k_s "0.05" \
    --max_reward_distance "0.0" \
    --waypoint_dist_tol "0.05"
