# 创建实验文件夹
exper_dir="${exper_base_dir}/${exper_name}"
mkdir -p ${exper_dir} 
# 创建checkpoints文件夹
checkpoints_dir="${exper_dir}/model"
mkdir -p ${checkpoints_dir}
# 创建maskspectrum文件夹
mask_spectrum_dir="${exper_dir}/maskspectrum"
mkdir -p ${mask_spectrum_dir}
# 创建日志文件夹
log_dir="${exper_dir}/log/${spot_id}"
mkdir -p ${log_dir}
# 创建评估结果文件夹
eva_dir="${exper_dir}/eva"
mkdir -p ${eva_dir}
# 创建预测结果数据文件夹
res_dir="${exper_dir}/res"
mkdir -p ${res_dir}