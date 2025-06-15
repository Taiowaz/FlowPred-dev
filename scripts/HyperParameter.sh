#!/bin/bash

run_file="Koopa/run.py"

is_training=0

if [ $is_training -eq 1 ]; then
    log_dir="Koopa/log/HyperParameter/train"
    root_base_path="data/0411/ogn/24/288"
else
    log_dir="Koopa/log/HyperParameter/test"
    root_base_path="data/0411/ogn/25/288"
fi

mkdir -p $log_dir

# dynamic_dim
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --dynamic_dim 256 --root_path "$root_base_path/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/dynamic_dim_256_mode0.log 2>&1 &
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --dynamic_dim 256 --root_path "$root_base_path/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/dynamic_dim_256_mode1.log 2>&1 &

# hidden_dim
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --hidden_dim 128 --root_path "$root_base_path/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/hidden_dim_128_mode0.log 2>&1 &
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --hidden_dim 128 --root_path "$root_base_path/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/hidden_dim_128_mode1.log 2>&1 &

# hidden_layers
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --hidden_layers 3 --root_path "$root_base_path/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/hidden_layers_3_mode0.log 2>&1 &
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --hidden_layers 3 --root_path "$root_base_path/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/hidden_layers_3_mode1.log 2>&1 &

# num_blocks
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --num_blocks 4 --root_path "$root_base_path/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/num_blocks_4_mode0.log 2>&1 &
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --num_blocks 4 --root_path "$root_base_path/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/num_blocks_4_mode1.log 2>&1 &

# alpha
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --alpha 0.3 --root_path "$root_base_path/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/alpha_0.3_mode0.log 2>&1 &
nohup /home/handb/.conda/envs/koopa/bin/python $run_file --is_training $is_training --alpha 0.3 --root_path "$root_base_path/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu 2 > $log_dir/alpha_0.3_mode1.log 2>&1 &