import os
import shutil

exper_name="debug_test_his24h_pred6h"
his_pred_dir_name = exper_name.split("_", 1)[1]
maskspectrum_dir = f"exper/{exper_name}/maskspectrum/{his_pred_dir_name}"
checkpoint_dir = f"exper/{exper_name}/checkpoint/{his_pred_dir_name}"


aux_dir = "aux_data"
aux_maskspectrum_dir = f"{aux_dir}/maskspectrum"
aux_checkpoint_dir = f"{aux_dir}/checkpoint"



# 处理maskspectrum目录
target_dir = os.path.join(aux_maskspectrum_dir, his_pred_dir_name)

# 判断目标路径是否存在
if os.path.exists(target_dir):
    # 遍历aux_maskspectrum_dir下的所有子目录
    for spot_dir in os.listdir(maskspectrum_dir):
        spot_path = os.path.join(maskspectrum_dir, spot_dir)
        target_spot_path = os.path.join(target_dir, spot_dir)
        if os.path.exists(target_spot_path):
            shutil.rmtree(target_spot_path)  
        shutil.copytree(spot_path, os.path.join(target_dir))
else:
    shutil.copytree(maskspectrum_dir, aux_maskspectrum_dir)

# 处理checkpoint目录
target_dir = os.path.join(aux_checkpoint_dir, his_pred_dir_name)
if os.path.exists(target_dir):
    # 遍历aux_checkpoint_dir下的所有子目录
    for spot_dir in os.listdir(checkpoint_dir):
        spot_path = os.path.join(checkpoint_dir, spot_dir)
        target_spot_path = os.path.join(target_dir, spot_dir)
        if os.path.exists(target_spot_path):
            shutil.rmtree(target_spot_path)  
        shutil.copytree(spot_path, os.path.join(target_dir))
else:
    shutil.copytree(checkpoint_dir, aux_checkpoint_dir)
