import pandas as pd
import os
os.chdir("/home/beihang/xihu/HZTourism/FlowPred-dev")
import sys
sys.path.append("/home/beihang/xihu/HZTourism/FlowPred-dev")
from pattern.pattern_train import get_group_annotation, save_mode_data
from tqdm import tqdm

df_dir = "data/pred6h_lanuch/raw/proc"

# 获取所有文件列表并排序
# df_files = sorted(os.listdir(df_dir))
# df_files = ["14208_240719_250720.csv"]
df_files = ["14207_240719_250720.csv"]



# 强制刷新输出函数
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

flush_print(f"Found {len(df_files)} files to process")
flush_print(f"Files: {df_files}")

# 使用tqdm添加进度条
for df_file in tqdm(df_files, desc="Processing files"):
    try:
        spot_id = int(df_file.split('_')[0]) 
        his_len = 288
        pred_len = 72
        time_interval="5min"
        if spot_id in [14207, 14208]:
            his_len = 2880
            pred_len = 720
            time_interval="30s"
        
        flush_print(f"Processing spot_id: {spot_id}")
        
        data_save_basedir = f"data/pred6h_lanuch/input/{str(spot_id)}"
        df = pd.read_csv(os.path.join(df_dir, df_file))
        
        # 处理 mode 0
        flush_print(f"  - Getting group annotation for mode 0...")
        groups_mode_0, groups_mode_1 = get_group_annotation(his_len=his_len,pred_len=pred_len, df=df, time_interval=time_interval)
        
        flush_print(f"  - Saving mode 0 data...")
        save_mode_data(
            groups_mode=groups_mode_0,
            mode=0,
            his_len=his_len,
            pred_len=pred_len,
            data_basepath=data_save_basedir,
        )
        
        flush_print(f"  - Saving mode 1 data...")
        save_mode_data(
            groups_mode=groups_mode_1,
            mode=1,
            his_len=his_len,
            pred_len=pred_len,
            data_basepath=data_save_basedir,
        )
        
        flush_print(f"  ✓ Completed spot_id: {spot_id}")
        
    except Exception as e:
        flush_print(f"  ❌ Error processing {df_file}: {str(e)}")
        continue

flush_print("All files processed successfully!")