#!/bin/bash

# 获取当前时间并格式化为 YYYY-MM-DD_HH-MM-SS
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

# 定义日志文件名
log_file="log/pred_6h_args/${current_time}.log"

run_file="run_args.py"

his_len=288
pred_len=72
checkpoints_savepath="data/pred_6h_args/res/model"
run_python="/home/beihang/.conda/envs/koopa/bin/python"
# 分模式运行

# train
# mode_0
nohup ${run_python} ${run_file} > ${log_file} 2>&1 &
echo "Log file: $log_file"
