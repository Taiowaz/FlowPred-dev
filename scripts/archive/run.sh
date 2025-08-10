#!/bin/bash

# 获取当前时间并格式化为 YYYY-MM-DD_HH-MM-SS
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

# 定义日志文件名
log_file="log_${current_time}.log"

run_file="Koopa/run.py"

# 分模式运行

# # train
# # his_len = 288
# # mode_0
# log_savepath="Koopa/log/train_mode0_288_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 1 --root_path "data/0411/ogn/24/288/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "2" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # mode_1
# log_savepath="Koopa/log/train_mode1_288_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 1 --root_path "data/0411/ogn/24/288/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "2" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # his_len = 2016
# # mode_0
# log_savepath="Koopa/log/train_mode0_2016_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 1 --root_path "data/0411/ogn/24/2016/mode_0" --spot_id "14100" --mode "0" --seq_len "2016" --checkpoints "data/0411/res/model" --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # mode_1
# log_savepath="Koopa/log/train_mode1_2016_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 1 --root_path "data/0411/ogn/24/2016/mode_1" --spot_id "14100" --mode "1" --seq_len "2016" --checkpoints "data/0411/res/model" --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # test
# # his_len = 288
# # mode_0
# log_savepath="Koopa/log/test_mode0_288_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0 --root_path "data/0411/ogn/25/288/mode_0" --spot_id "14100" --mode "0" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "2" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"


# # mode_1
# log_savepath="Koopa/log/test_mode1_288_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0 --root_path "data/0411/ogn/25/288/mode_1" --spot_id "14100" --mode "1" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # his_len = 2016
# # mode_0
# log_savepath="Koopa/log/test_mode0_2016_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0 --root_path "data/0411/ogn/25/2016/mode_0" --spot_id "14100" --mode "0" --seq_len "2016" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"


# # mode_1
# log_savepath="Koopa/log/test_mode1_2016_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0 --root_path "data/0411/ogn/25/2016/mode_1" --spot_id "14100" --mode "1" --seq_len "2016" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"


# 不分模式
# # train
# log_savepath="Koopa/log/train_288_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 1  --root_path "data/0411/ogn/24/mode_no" --spot_id "14100" --mode "no" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # # test
# log_savepath="Koopa/log/test_288_mode0_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0  --root_path "data/0411/ogn/25/288/mode_no" --spot_id "14100" --mode "no" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # 测试不区分模式训练的模型 周末的预测效果
# log_savepath="Koopa/log/test_288_gen_mode0_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0  --root_path "data/0411/ogn/25/288/mode_0" --spot_id "14100" --mode "no" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # 测试不区分模式训练的模型 工作日的预测效果
# log_savepath="Koopa/log/test_288_gen_mode1_${log_file}"
# nohup /home/handb/.conda/envs/koopa/bin/python ${run_file} --is_training 0  --root_path "data/0411/ogn/25/288/mode_1" --spot_id "14100" --mode "no" --seq_len "288" --checkpoints "data/0411/res/model" --gpu "1" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"