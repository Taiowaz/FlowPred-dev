#!/bin/bash

run_python="/home/beihang/.conda/envs/koopa/bin/python"
run_file="src/run.py"
exper_base_dir="exper"
exper_data_base_dir="exper_data"
exper_name="6hlaunch_288_72"
spot_id="14207"
his_hour=24
pred_hour=6
his_len=2880
pred_len=720

# 执行前置脚本
source /home/beihang/xihu/HZTourism/FlowPred-dev/scripts/template/prefix.sh

# train
data_multi_basedir="${exper_data_base_dir}/${exper_name}/train"

mode="0"
log_savepath="${log_dir}/train_mode${mode}.log"
nohup ${run_python} ${run_file} \
    --is_training 1 \
    --data_multi_basedir "${data_multi_basedir}" \
    --mask_spectrum_dir "${mask_spectrum_dir}"\
    --spot_id ${spot_id} \
    --mode ${mode} \
    --seq_len ${his_len} \
    --label_len ${pred_len} \
    --pred_len ${pred_len} \
    --checkpoints ${checkpoints_dir} \
    --gpu "0" > ${log_savepath} 2>&1 &
echo "Log file: $log_savepath"

mode="1"
log_savepath="${log_dir}/train_mode${mode}.log"
nohup ${run_python} ${run_file} \
    --is_training 1 \
    --data_multi_basedir "${data_multi_basedir}" \
    --mask_spectrum_dir "${mask_spectrum_dir}"\
    --spot_id ${spot_id} \
    --mode ${mode} \
    --his_hour ${his_hour} \
    --pred_hour ${pred_hour} \
    --seq_len ${his_len} \
    --label_len ${pred_len} \
    --pred_len ${pred_len} \
    --checkpoints ${checkpoints_dir} \
    --gpu "0" > ${log_savepath} 2>&1 &

echo "Log file: $log_savepath"



# # test
# data_base_dir="${exper_data_base_dir}/${exper_name}/test/${spot_id}"
# res_dir="${exper_data_base_dir}/${exper_name}/res"

# mode="0"
# log_savepath="${log_dir}/test_mode${mode}.log"
# nohup ${run_python} ${run_file} \
#     --is_training 0 \
#     --root_path "${data_base_dir}/mode${mode}" \
#     --mask_spectrum_dir "${mask_spectrum_dir}"\
#     --test_res_save_dir "${res_dir}" \
#     --spot_id "${spot_id}" \
#     --mode ${mode} \
#     --seq_len ${his_len} \
#     --label_len ${pred_len} \
#     --pred_len ${pred_len} \
#     --checkpoints ${checkpoints_dir} \
#     --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"

# mode="1"
# log_savepath="${log_dir}/test_mode${mode}.log"
# nohup ${run_python} ${run_file} \
#     --is_training 0 \
#     --root_path "${data_base_dir}/mode${mode}" \
#     --mask_spectrum_dir "${mask_spectrum_dir}"\
#     --test_res_save_dir "${res_dir}" \
#     --spot_id "${spot_id}" \
#     --mode ${mode} \
#     --seq_len ${his_len} \
#     --label_len ${pred_len} \
#     --pred_len ${pred_len} \
#     --checkpoints ${checkpoints_dir} \
#     --gpu "0" > ${log_savepath} 2>&1 &
# echo "Log file: $log_savepath"
