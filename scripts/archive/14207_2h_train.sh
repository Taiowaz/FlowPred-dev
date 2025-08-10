#!/bin/bash

# 获取当前时间并格式化为 YYYY-MM-DD_HH-MM-SS
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

# 定义日志文件名
log_file="log_${current_time}.log"

run_file="run.py"

his_len=288
pred_len=24
checkpoints_savepath="data/14207/res/model"
run_python="/home/beihang/.conda/envs/koopa/bin/python"
# 分模式运行

# train
# mode_0
log_savepath="log/pred_6h/train_mode0_${his_len}_${pred_len}_${log_file}"
nohup ${run_python} ${run_file} --is_training 1 --root_path "data/14207/ogn/train/288_24/mode_0" --spot_id "14100" --mode "0" --seq_len ${his_len} --pred_len ${pred_len} --checkpoints ${checkpoints_savepath} --gpu "0" > ${log_savepath} 2>&1 &
echo "Log file: $log_savepath"

# mode_1
log_savepath="log/pred_6h/train_mode1_${his_len}_${pred_len}_${log_file}"
nohup ${run_python} ${run_file} --is_training 1 --root_path "data/14207/ogn/train/288_24/mode_1" --spot_id "14100" --mode "1" --seq_len ${his_len} --pred_len ${pred_len} --checkpoints ${checkpoints_savepath} --gpu "0" > ${log_savepath} 2>&1 &
echo "Log file: $log_savepath"