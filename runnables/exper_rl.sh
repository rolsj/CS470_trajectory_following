#!/bin/bash

if [ $# -ne 4 ]
  then
    echo "Please supply mode, h1, h2, l as an argument!"
    echo "Usage) vis_rl.sh auto 1 1 10"
    echo "Usage) vis_rl.sh drive 0.5 0.5 7"
    echo "Usage) vis_rl.sh flight 0.5 0.7 5"
    exit -1
fi

echo "mode=$1"
echo "h1=$2"
echo "h2=$3"
echo "l=$4"

python3 ../agents/C_trajectory_following.py \
    --gui "False" \
    --output_folder "../checkpointed_models" \
    --timesteps 2500000 \
    --train "False" \
    --test "False" \
    --vis "True" \
    --n_envs "5" \
    --episode_len_sec "60" \
    --waypoint_buffer_size "2" \
    --k_p "5.0" \
    --k_wp "8.0" \
    --k_s "0.05" \
    --max_reward_distance "0.0" \
    --waypoint_dist_tol "0.05" \
    --rand "False" \
    --n "1" \
    --mode $1 \
    --h1 $2 \
    --h2 $3 \
    --l $4
