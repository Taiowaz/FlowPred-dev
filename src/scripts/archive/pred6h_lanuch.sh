#!/bin/bash

exper_name="pred6h_lanuch"
mask_spectrum_dir="data/${exper_name}/maskspectrum"
checkpoints_savepath="data/${exper_name}/model"
run_python="/home/beihang/.conda/envs/koopa/bin/python"
run_file="run.py"

# 景点ID列表
spot_ids=("14100" "14102" "14103" "14105" "14107" "14108" "14114" "14115" "14116" "14120" "14124" "14125" "14126" "14127" "14129" "14137" "14141" "14144" "14145" "14205" "14207" "14208")

for spot_id in "${spot_ids[@]}"; do

    his_len=288
    pred_len=72
    if [[ "$spot_id" == "14207" || "$spot_id" == "14208" ]]; then
        his_len=2880
        pred_len=720
    fi

    log_file="log/${exper_name}/${spot_id}_0_train.log"
    $run_python $run_file --is_training 1 \
        --root_path "data/${exper_name}/input/${spot_id}/${his_len}_${pred_len}/mode_0" \
        --spot_id ${spot_id} \
        --mode "0" \
        --seq_len ${his_len} \
        --pred_len ${pred_len} \
        --mask_spectrum_dir ${mask_spectrum_dir} \
        --checkpoints ${checkpoints_savepath} \
        --gpu "0" > $log_file 2>&1
    echo "Log file: $log_file"

    log_file="log/${exper_name}/${spot_id}_1_train.log"
    $run_python $run_file --is_training 1 \
        --root_path "data/${exper_name}/input/${spot_id}/${his_len}_${pred_len}/mode_1" \
        --spot_id ${spot_id} \
        --mode "1" \
        --seq_len ${his_len} \
        --pred_len ${pred_len} \
        --mask_spectrum_dir ${mask_spectrum_dir} \
        --checkpoints ${checkpoints_savepath} \
        --gpu "0" > $log_file 2>&1
    echo "Log file: $log_file"
done