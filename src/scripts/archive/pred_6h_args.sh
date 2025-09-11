#!/bin/bash

# 获取当前时间并格式化为 YYYY-MM-DD_HH-MM-SS
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

# 定义日志文件名
log_file="log/pred_6h_args/${current_time}.log"


his_len=288
pred_len=72
checkpoints_savepath="data/pred_6h_args/res/model"
run_python="/home/beihang/.conda/envs/koopa/bin/python"
# 分模式运行

# # train
# # 模式以及相关路径在run_args.py中定义
# run_file="run_args.py"
# nohup ${run_python} ${run_file} > ${log_file} 2>&1 &
# echo "Log file: $log_file"

# 默认参数训练
# log_file="log/pred_6h_args/train_default_${current_time}.log"
# run_file="run.py"
# nohup $run_python $run_file --is_training 1 \
#     --root_path "data/pred_6h_args/ogn/24/288_72/mode_0" \
#     --spot_id "14100" \
#     --mode "0" \
#     --seq_len ${his_len} \
#     --pred_len ${pred_len} \
#     --checkpoints ${checkpoints_savepath} \
#     --gpu "0" > $log_file 2>&1 &
# echo "Log file: $log_file"
# log_file="log/pred_6h_args/train_default_${current_time}.log"
# run_file="run.py"
# nohup $run_python $run_file --is_training 1 \
#     --root_path "data/pred_6h_args/ogn/24/288_72/mode_1" \
#     --spot_id "14100" \
#     --mode "1" \
#     --seq_len ${his_len} \
#     --pred_len ${pred_len} \
#     --checkpoints ${checkpoints_savepath} \
#     --gpu "0" > $log_file 2>&1 &
# echo "Log file: $log_file"



# test 最优参数
# 节假日
# log_file="log/pred_6h_args/test_mode0_${current_time}.log"
# run_file="run.py"
# nohup $run_python $run_file --is_training 0 \
#     --root_path "data/pred_6h_args/ogn/25/288_72/mode_0" \
#     --spot_id "14100" \
#     --mode "0" \
#     --seq_len "288" \
#     --label_len "24" \
#     --pred_len "72" \
#     --checkpoints "data/pred_6h_args/res/model" \
#     --dynamic_dim 128 \
#     --hidden_dim 256 \
#     --hidden_layers 4 \
#     --num_blocks 2 \
#     --alpha 0.1 \
#     --gpu 0 > $log_file 2>&1 &

# log_file="log/pred_6h_args/test_mode1_${current_time}.log"
# nohup $run_python $run_file --is_training 0 \
#     --root_path "data/pred_6h_args/ogn/25/288_72/mode_1" \
#     --spot_id "14100" \
#     --mode "1" \
#     --seq_len "288" \
#     --label_len "24" \
#     --pred_len "72" \
#     --checkpoints "data/pred_6h_args/res/model" \
#     --dynamic_dim 256 \
#     --hidden_dim 256 \
#     --hidden_layers 4 \
#     --num_blocks 5 \
#     --alpha 0.2 \
#     --gpu 0 > $log_file 2>&1 &




# test 默认参数
log_file="log/pred_6h_args/test_mode0_default_${current_time}.log"
run_file="run.py"
nohup $run_python $run_file --is_training 0 \
    --root_path "data/pred_6h_args/ogn/25/288_72/mode_0" \
    --spot_id "14100" \
    --mode "0" \
    --seq_len "288" \
    --label_len "24" \
    --pred_len "72" \
    --checkpoints "data/pred_6h_args/res/model" \
    --dynamic_dim 128 \
    --hidden_dim 64 \
    --hidden_layers 2 \
    --num_blocks 3 \
    --alpha 0.2 \
    --gpu 0 > $log_file 2>&1 &

log_file="log/pred_6h_args/test_mode1_default_${current_time}.log"
run_file="run.py"
nohup $run_python $run_file --is_training 0 \
    --root_path "data/pred_6h_args/ogn/25/288_72/mode_1" \
    --spot_id "14100" \
    --mode "1" \
    --seq_len "288" \
    --label_len "24" \
    --pred_len "72" \
    --checkpoints "data/pred_6h_args/res/model" \
    --dynamic_dim 128 \
    --hidden_dim 64 \
    --hidden_layers 2 \
    --num_blocks 3 \
    --alpha 0.2 \
    --gpu 0 > $log_file 2>&1 &

# 输出pid到run.pid
echo $! > run.pid

