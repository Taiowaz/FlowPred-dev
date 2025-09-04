import os
os.chdir("/home/beihang/xihu/HZTourism/FlowPred-dev")
import sys
sys.path.append("/home/beihang/xihu/HZTourism/FlowPred-dev")
from src.utils.utils_data import get_spot_config
from src.utils.utils_eva_db import save_csv_from_db
import pandas as pd
from src.pattern.pattern_train import get_group_annotation, save_mode_data

his_hour=24
pred_hour=6

# 景点ID列表，5min
# 1开头
# spot_ids = [14100,14102,14103,14104,14105,14107,14108,14114,14115,14116,14120,14124,14125,14126,14127,14129,14137,14141,14144,14145,14205]
# 2开头
spot_ids = [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019, 20020, 20021, 20022, 20023, 20024, 20025]

print(f"=== 开始处理训练数据预处理 ===")
print(f"历史时长: {his_hour}小时, 预测时长: {pred_hour}小时")
print(f"总共需要处理 {len(spot_ids)} 个景点: {spot_ids}")

# 配置项
exper_name = f"mse_loss_his{his_hour}h-pred{pred_hour}h"
exper_dir = f"exper/{exper_name}"
exper_data_dir = f"exper_data/{exper_name}"
os.makedirs(exper_data_dir, exist_ok=True)
print(f"实验目录: {exper_data_dir}")

for idx, spot_id in enumerate(spot_ids, 1):
    print(f"\n【{idx}/{len(spot_ids)}】正在处理景点 {spot_id}...")

    raw_dir = f"{exper_data_dir}/raw/{spot_id}"
    proc_dir = f"{exper_data_dir}/proc/{spot_id}"
    train_dir = f"{exper_data_dir}/train/{spot_id}"
    test_dir = f"{exper_data_dir}/test/{spot_id}"
    res_dir = f"{exper_data_dir}/res/{spot_id}"

    print(f"  - 创建目录结构...")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)   
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    freq, his_len, pred_len = get_spot_config(spot_id, his_hour, pred_hour)
    print(f"  - 景点配置: 频率={freq}, 历史长度={his_len}, 预测长度={pred_len}")

    s_time = "2024-07-20"
    e_time = "2025-07-20"
    if e_time is None:
        file_base_name = f"{spot_id}_{s_time}"
        train_raw_data_file=f"{raw_dir}/{file_base_name}.csv"
    else:
        file_base_name = f"{spot_id}_{s_time}_{e_time}"
        train_raw_data_file=f"{raw_dir}/{file_base_name}.csv"
    
    print(f"  - 从数据库获取原始数据 ({s_time} 到 {e_time})...")
    save_csv_from_db(
        spot_id=spot_id,
        s_time=f"{s_time} 00:00:00",
        e_time= f"{e_time} 23:59:59",
        output_csv_file=train_raw_data_file,
    )

    print(f"  - 读取并预处理数据...")
    df = pd.read_csv(train_raw_data_file)
    print(f"    原始数据行数: {len(df)}")
    
    # 按kpi_time列转换为datetime格式
    df['kpi_time'] = pd.to_datetime(df['kpi_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # 按kpi_time去重
    df = df.drop_duplicates(subset=['kpi_time'])
    print(f"    去重后数据行数: {len(df)}")
    
    # 按kpi_time排序
    df = df.sort_values(by='kpi_time')
    
    # 数据预处理，分景点
    print(f"  - 根据景点类型进行特定预处理...")
    if spot_id in [14210,14211,14212,14213]:
        print(f"    景点 {spot_id} 跳过预处理")
        pass
    elif spot_id in [14207,14209]:
        print(f"    景点 {spot_id} 使用30秒频率预处理")
        from src.utils.utils_data import fill_missing_value_singlespot_30s,preprocess_for_koopman_30s_moderate
        df_proc = fill_missing_value_singlespot_30s(spot_id, df)
        # 添加Koopman专用预处理
        df_proc = preprocess_for_koopman_30s_moderate(df_proc, spot_id)
    elif spot_id in [14208]:
        print(f"    景点 {spot_id} 跳过预处理")
        pass
    else:
        print(f"    景点 {spot_id} 使用日频率预处理")
        from src.utils.utils_data import fill_missing_value_singlespot_day
        df_proc = fill_missing_value_singlespot_day(df, freq=freq)

    print(f"    预处理后数据行数: {len(df_proc)}")
    df_proc.to_csv(f"{proc_dir}/{file_base_name}_proc.csv", index=False)
    print(f"  - 保存预处理数据到: {proc_dir}/{file_base_name}_proc.csv")

    print(f"  - 生成训练数据组...")
    df_proc = pd.read_csv(f"{proc_dir}/{file_base_name}_proc.csv")
    save_base_dir = train_dir
    os.makedirs(save_base_dir, exist_ok=True)
    groups_mode_0, groups_mode_1 = get_group_annotation(his_len=his_len,pred_len=pred_len, df=df_proc, time_interval=freq)
    
    print(f"    Mode 0 数据组数量: {len(groups_mode_0)}")
    save_mode_data(
        groups_mode=groups_mode_0,
        mode=0,
        data_basepath=save_base_dir,
    )
    
    print(f"    Mode 1 数据组数量: {len(groups_mode_1)}")
    save_mode_data(
        groups_mode=groups_mode_1,
        mode=1,
        data_basepath=save_base_dir,
    )
    
    print(f"  ✓ 景点 {spot_id} 处理完成")

print(f"\n=== 所有景点处理完成！===")
print(f"共处理了 {len(spot_ids)} 个景点")
print(f"数据保存在: {exper_data_dir}")