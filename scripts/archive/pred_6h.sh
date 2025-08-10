#!/bin/bash

# 获取当前时间并格式化为 YYYY-MM-DD_HH-MM-SS
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

# 定义日志文件名
log_file="log_${current_time}.log"

run_file="run.py"

his_len=288
pred_len=72
checkpoints_savepath="data/pred_6h/res/model"
run_python="/home/beihang/.conda/envs/koopa/bin/python"
test_res_save_dir="data/pred_6h/res/res_test"
# 分模式运行

# # train
# # mode_0
# log_savepath="log/pred_6h/train_mode0_${his_len}_${pred_len}_${log_file}"
# nohup ${run_python} ${run_file} --is_training 1 --root_path "data/pred_6h/ogn/24/288_72/mode_0" --spot_id "14100" --mode "0" --seq_len ${his_len} --pred_len ${pred_len} --checkpoints ${checkpoints_savepath} --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # mode_1
# log_savepath="log/pred_6h/train_mode1_${his_len}_${pred_len}_${log_file}"
# nohup ${run_python} ${run_file} --is_training 1 --root_path "data/pred_6h/ogn/24/288_72/mode_1" --spot_id "14100" --mode "1" --seq_len ${his_len} --pred_len ${pred_len} --checkpoints ${checkpoints_savepath} --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# # test
# # mode_0
# log_savepath="log/pred_6h/test_mode0_${his_len}_${pred_len}_${log_file}"
# nohup ${run_python} ${run_file} --is_training 0 --root_path "data/pred_6h/ogn/25/288_72/mode_0" --test_res_save_dir ${test_res_save_dir} --spot_id "14100" --mode "0" --seq_len "288" --pred_len "72" --checkpoints ${checkpoints_savepath} --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"


# # mode_1
# log_savepath="log/pred_6h/test_mode1_${his_len}_${pred_len}_${log_file}"
# nohup ${run_python} ${run_file} --is_training 0 --root_path "data/pred_6h/ogn/25/288_72/mode_1" --test_res_save_dir ${test_res_save_dir} --spot_id "14100" --mode "1" --seq_len "288" --pred_len "72" --checkpoints ${checkpoints_savepath} --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"
