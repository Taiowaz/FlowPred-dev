import os
import shutil

exper_name="mse_loss_his24h-pred6h"
his_pred_dir_name = exper_name.split("_")[-1]
maskspectrum_dir = f"exper/{exper_name}/maskspectrum/{his_pred_dir_name}"
checkpoint_dir = f"exper/{exper_name}/checkpoint/{his_pred_dir_name}"

print(f"实验名称: {exper_name}")
print(f"历史预测目录名: {his_pred_dir_name}")
print(f"源maskspectrum目录: {maskspectrum_dir}")
print(f"源checkpoint目录: {checkpoint_dir}")

aux_dir = "aux_data"
aux_maskspectrum_dir = f"{aux_dir}/maskspectrum"
aux_checkpoint_dir = f"{aux_dir}/checkpoint"

print(f"辅助数据目录: {aux_dir}")
print(f"辅助maskspectrum目录: {aux_maskspectrum_dir}")
print(f"辅助checkpoint目录: {aux_checkpoint_dir}")

# 处理maskspectrum目录
target_dir = os.path.join(aux_maskspectrum_dir, his_pred_dir_name)

print(f"\n=== 处理maskspectrum目录 ===")
print(f"目标目录: {target_dir}")
print(f"源目录是否存在: {os.path.exists(maskspectrum_dir)}")
print(f"目标目录是否存在: {os.path.exists(target_dir)}")

# 判断目标路径是否存在
if os.path.exists(target_dir):
    print("目标目录已存在，逐个复制子目录...")
    # 遍历aux_maskspectrum_dir下的所有子目录
    for spot_dir in os.listdir(maskspectrum_dir):
        spot_path = os.path.join(maskspectrum_dir, spot_dir)
        target_spot_path = os.path.join(target_dir, spot_dir)
        print(f"  处理子目录: {spot_dir}")
        print(f"    源路径: {spot_path}")
        print(f"    目标路径: {target_spot_path}")
        if os.path.exists(target_spot_path):
            print(f"    删除已存在的目标路径: {target_spot_path}")
            shutil.rmtree(target_spot_path)  
        print(f"    复制目录...")
        shutil.copytree(spot_path, os.path.join(target_dir, spot_dir))
else:
    print("目标目录不存在，直接复制整个目录...")
    # 确保父目录存在
    os.makedirs(aux_maskspectrum_dir, exist_ok=True)
    shutil.copytree(maskspectrum_dir, target_dir)

# 处理checkpoint目录
target_dir = os.path.join(aux_checkpoint_dir, his_pred_dir_name)

print(f"\n=== 处理checkpoint目录 ===")
print(f"目标目录: {target_dir}")
print(f"源目录是否存在: {os.path.exists(checkpoint_dir)}")
print(f"目标目录是否存在: {os.path.exists(target_dir)}")

if os.path.exists(target_dir):
    print("目标目录已存在，逐个复制子目录...")
    # 遍历aux_checkpoint_dir下的所有子目录
    for spot_dir in os.listdir(checkpoint_dir):
        spot_path = os.path.join(checkpoint_dir, spot_dir)
        target_spot_path = os.path.join(target_dir, spot_dir)
        print(f"  处理子目录: {spot_dir}")
        print(f"    源路径: {spot_path}")
        print(f"    目标路径: {target_spot_path}")
        if os.path.exists(target_spot_path):
            print(f"    删除已存在的目标路径: {target_spot_path}")
            shutil.rmtree(target_spot_path)  
        print(f"    复制目录...")
        # 目标路径必须不存在
        shutil.copytree(spot_path, os.path.join(target_dir, spot_dir))
else:
    print("目标目录不存在，直接复制整个目录...")
    # 确保父目录存在
    os.makedirs(aux_checkpoint_dir, exist_ok=True)
    shutil.copytree(checkpoint_dir, target_dir)

print("\n复制操作完成！")