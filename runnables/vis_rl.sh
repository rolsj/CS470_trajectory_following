#!/bin/bash

if [ $# -ne 3 ]
  then
    echo "Please supply h1, h2, l as an argument!"
    echo "Usage) vis_rl.sh 1 1 5"
    exit -1
fi

echo "h1=$1"
echo "h2=$2"
echo "l=$3"

python ../agents/C_trajectory_following.py \
    --gui "True" \
    --output_folder "../checkpointed_models" \
    --timesteps 2500000 \
    --train "False" \
    --test "False" \
    --vis "True" \
    --n_envs "5" \
    --episode_len_sec "20" \
    --waypoint_buffer_size "2" \
    --k_p "5.0" \
    --k_wp "8.0" \
    --k_s "0.05" \
    --max_reward_distance "0.0" \
    --waypoint_dist_tol "0.05" \
    --h1 $1 \
    --h2 $2 \
    --l $3
