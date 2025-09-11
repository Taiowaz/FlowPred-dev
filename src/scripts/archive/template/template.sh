#!/bin/bash

run_python="/home/beihang/.conda/envs/koopa/bin/python"
run_file="src/run.py"
exper_base_dir="exper"
exper_name="template" 
data_base_dir="exper_data/${exper_name}"
# spot_ids=("14100" "14102" "14103" "14105" "14107" "14108" "14114" "14115" "14116" "14120" "14124" "14125" "14126" "14127" "14129" "14137" "14141" "14144" "14145" "14205" "14207" "14208")
spot_ids=("14100" "14207")

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $1"
}

# 脚本开始
log_info "==========================================="
log_info "开始执行训练脚本"
log_info "实验名称: ${exper_name}"
log_info "数据目录: ${data_base_dir}"
log_info "Python路径: ${run_python}"
log_info "运行文件: ${run_file}"
log_info "==========================================="

# 执行前置脚本
# source /home/beihang/xihu/HZTourism/FlowPred-dev/scripts/template/prefix.sh

exper_dir="${exper_base_dir}/${exper_name}"
log_info "创建实验目录: ${exper_dir}"
mkdir -p ${exper_dir} 

# 创建checkpoints文件夹
checkpoints_dir="${exper_dir}/model"
log_info "创建模型保存目录: ${checkpoints_dir}"
mkdir -p ${checkpoints_dir}

# 创建maskspectrum文件夹
mask_spectrum_dir="${exper_dir}/maskspectrum"
log_info "创建mask spectrum目录: ${mask_spectrum_dir}"
mkdir -p ${mask_spectrum_dir}

# 创建评估结果文件夹
eva_dir="${exper_dir}/eva"
log_info "创建评估结果目录: ${eva_dir}"
mkdir -p ${eva_dir}

# 创建预测结果数据文件夹
res_dir="${exper_dir}/res"
log_info "创建预测结果目录: ${res_dir}"
mkdir -p ${res_dir}


log_info "景点ID列表: ${spot_ids[*]}"
log_info "总共需要处理 ${#spot_ids[@]} 个景点"

# 统计变量
total_tasks=0
started_tasks=0

for spot_id in "${spot_ids[@]}"; do
    log_info "-------------------------------------------"
    log_info "开始处理景点ID: ${spot_id}"
    
    # 创建该景点的日志文件夹
    log_dir="${exper_dir}/log/${spot_id}"
    log_info "创建景点日志目录: ${log_dir}"
    mkdir -p ${log_dir}
    
    # 根据spot_id设置频率和长度参数
    if [[ "$spot_id" == "14210" || "$spot_id" == "14211" || "$spot_id" == "14212" || "$spot_id" == "14213" ]]; then
        freq="10min"
        his_len=144
        pred_len=36
        log_info "景点 ${spot_id} 使用10分钟频率配置"
    elif [[ "$spot_id" == "14207" || "$spot_id" == "14209" ]]; then
        freq="30sec"
        his_len=2880
        pred_len=720
        log_info "景点 ${spot_id} 使用30秒频率配置"
    elif [[ "$spot_id" == "14208" ]]; then
        freq="1min"
        his_len=1440
        pred_len=360
        log_info "景点 ${spot_id} 使用1分钟频率配置"
    else
        freq="5min"
        his_len=288
        pred_len=72
        log_info "景点 ${spot_id} 使用5分钟频率配置(默认)"
    fi
    
    log_info "景点参数 -> 频率: ${freq}, 历史长度: ${his_len}, 预测长度: ${pred_len}"

    # 检查数据目录是否存在
    if [[ ! -d "${data_base_dir}/mode_0" ]]; then
        log_warning "数据目录不存在: ${data_base_dir}/mode_0"
    fi
    if [[ ! -d "${data_base_dir}/mode_1" ]]; then
        log_warning "数据目录不存在: ${data_base_dir}/mode_1"
    fi

    # 模式0训练
    mode="0"
    log_savepath="${log_dir}/train_mode${mode}.log"
    total_tasks=$((total_tasks + 1))
    
    log_info "启动模式${mode}训练任务..."
    log_info "日志文件: ${log_savepath}"
    
    nohup ${run_python} ${run_file} \
        --is_training 1 \
        --root_path "${data_base_dir}/mode_${mode}" \
        --mask_spectrum_dir "${mask_spectrum_dir}"\
        --spot_id ${spot_id} \
        --mode ${mode} \
        --freq ${freq} \
        --seq_len ${his_len} \
        --label_len ${pred_len} \
        --pred_len ${pred_len} \
        --checkpoints ${checkpoints_dir} \
        --gpu "0" > ${log_savepath} 2>&1 &
    
    mode0_pid=$!
    started_tasks=$((started_tasks + 1))
    log_info "模式${mode}训练任务已启动, PID: ${mode0_pid}"

    # 模式1训练
    mode="1"
    log_savepath="${log_dir}/train_mode${mode}.log"
    total_tasks=$((total_tasks + 1))
    
    log_info "启动模式${mode}训练任务..."
    log_info "日志文件: ${log_savepath}"
    
    nohup ${run_python} ${run_file} \
        --is_training 1 \
        --root_path "${data_base_dir}/mode_${mode}" \
        --mask_spectrum_dir "${mask_spectrum_dir}"\
        --spot_id ${spot_id} \
        --mode ${mode} \
        --freq ${freq} \
        --seq_len ${his_len} \
        --label_len ${pred_len} \
        --pred_len ${pred_len} \
        --checkpoints ${checkpoints_dir} \
        --gpu "0" > ${log_savepath} 2>&1 &
    
    mode1_pid=$!
    started_tasks=$((started_tasks + 1))
    log_info "模式${mode}训练任务已启动, PID: ${mode1_pid}"
    
    log_info "景点 ${spot_id} 的两个模式训练任务都已启动"
    
    # 添加短暂延迟避免GPU资源冲突
    sleep 2

done

log_info "==========================================="
log_info "所有训练任务启动完成!"
log_info "总任务数: ${total_tasks}"
log_info "成功启动: ${started_tasks}"
log_info "可以使用以下命令监控训练进度:"
log_info "  - 查看所有训练进程: ps aux | grep 'run.py'"
log_info "  - 查看GPU使用情况: nvidia-smi"
log_info "  - 查看日志: tail -f ${exper_dir}/log/<spot_id>/train_mode<mode>.log"
log_info "==========================================="

# 创建监控脚本
monitor_script="${exper_dir}/monitor.sh"
cat > ${monitor_script} << 'EOF'
#!/bin/bash
echo "=== 训练进程监控 ==="
echo "当前运行的训练进程:"
ps aux | grep "run.py" | grep -v grep
echo ""
echo "=== GPU使用情况 ==="
nvidia-smi
echo ""
echo "=== 最新日志 ==="
find exper/template/log -name "*.log" -exec echo "=== {} ===" \; -exec tail -3 {} \;
EOF

chmod +x ${monitor_script}
log_info "创建了监控脚本: ${monitor_script}"
log_info "运行 ${monitor_script} 可以查看训练状态"