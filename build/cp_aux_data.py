import os
import shutil


ogn_base_path = "/home/beihang/xihu/HZTourism/FlowPred-dev/data/pred6h_lanuch"
pur_base_path = "/home/www/FlowPred39"

# name = "model"
# for spot in os.listdir(os.path.join(ogn_base_path, name)):
#     pred_name = "288_72"
#     if spot in ["14207", "14208"]:
#         pred_name = "2880_720"

#     f = os.path.join(ogn_base_path, name, spot, pred_name)
#     pur_dir = os.path.join(pur_base_path, "modelbase", spot)
#     dest_path = os.path.join(pur_dir, pred_name)
    
#     if os.path.exists(f):
#         print(f"Processing {f}...")
#     # 如果目标路径已存在，先删除
#     if os.path.exists(dest_path):
#         shutil.rmtree(dest_path)
    
#     # 复制整个文件夹
#     shutil.copytree(f, dest_path)
#     print(f"Copied {f} to {dest_path}")


# 修改文件参数名
pur_base_path  = "/home/www/FlowPred39/modelbase"
for spot in os.listdir(pur_base_path):
    pred_name = "288_72"
    if spot in ["14207", "14208"]:
        pred_name = "2880_720"

    pur_dir = os.path.join(pur_base_path, spot, pred_name)
    if os.path.exists(pur_dir):
        path0 = os.path.join(pur_dir, "0")
        path1 = os.path.join(pur_dir, "1")
        for f in os.listdir(path0):
            # 修改文件名
            seq_len = pred_name.split("_")[0]
            pred_len = pred_name.split("_")[1]
            new_name = f"{seq_len}_24_{pred_len}_dyna128_h64_l2_nb3_a0.2"
            os.rename(os.path.join(path0, f), os.path.join(path0, new_name))
            print(f"Renamed {f} to {new_name} in {path0}")
        for f in os.listdir(path1):
            # 修改文件名
            seq_len = pred_name.split("_")[0]
            pred_len = pred_name.split("_")[1]
            new_name = f"{seq_len}_24_{pred_len}_dyna128_h64_l2_nb3_a0.2"
            os.rename(os.path.join(path1, f), os.path.join(path1, new_name))
            print(f"Renamed {f} to {new_name} in {path1}")
