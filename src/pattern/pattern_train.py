import pandas as pd
import numpy as np
import os
import holidays
from tqdm import tqdm
import gc

def get_group_annotation(df, his_len=288, pred_len=24, time_interval="5min"):
    print("Starting group annotation...")
    
    # 获取连续性分组，初步按照时间间隔划分
    print("Getting continuous groups...")
    all_groups = get_continuous_group(df, time_interval)
    print(f"Found {len(all_groups)} continuous groups")
    
    # 创建中国节假日对象
    china_holidays = holidays.China()

    # 初始化临时数据存储
    holiday_samples = []
    workday_samples = []
    
    # 预计算历史长度和预测长度总和
    total_len = his_len + pred_len
    
    # 批量处理参数
    BATCH_SIZE = 500  # 每批处理500个样本
    holiday_batch_dfs = []
    workday_batch_dfs = []
    
    print(f"Processing groups (his_len={his_len}, pred_len={pred_len})...")
    
    # 遍历每个连续性分组
    for group_idx, group in enumerate(tqdm(all_groups, desc="Processing continuous groups", unit="group")):
        # 如果分组长度不足以包含历史长度和预测长度，跳过该分组
        if len(group) < total_len:
            continue
        
        # 添加mode列（创建副本避免修改原数据）
        group = group.copy()
        group["mode"] = group["kpi_time"].apply(lambda x: 0 if x.date() in china_holidays or x.weekday() >= 5 else 1)

        # 遍历当前分组，按照历史长度和预测长度拆分样本
        sample_range = range(len(group) - total_len + 1)
        
        for i in sample_range:
            # 提取当前样本
            group_item = group.iloc[i: i + total_len].copy()
            
            # 统计预测数据段中的mode
            pred_modes = group_item["mode"].iloc[-pred_len:]
            num_mode_0 = np.sum(pred_modes == 0)
            num_mode_1 = pred_len - num_mode_0
            
            # 根据mode数量分配到对应模式
            if num_mode_0 > num_mode_1:
                holiday_samples.append(group_item)
            else:
                workday_samples.append(group_item)
            
            # 批量处理：当累积到一定数量时，合并并清空
            if len(holiday_samples) >= BATCH_SIZE:
                batch_df = pd.concat(holiday_samples, ignore_index=True)
                batch_df = batch_df.drop_duplicates(subset=["kpi_time"])
                holiday_batch_dfs.append(batch_df)
                holiday_samples.clear()  # 清空列表释放内存
                del batch_df
                gc.collect()
            
            if len(workday_samples) >= BATCH_SIZE:
                batch_df = pd.concat(workday_samples, ignore_index=True)
                batch_df = batch_df.drop_duplicates(subset=["kpi_time"])
                workday_batch_dfs.append(batch_df)
                workday_samples.clear()  # 清空列表释放内存
                del batch_df
                gc.collect()
    
    # 处理剩余的样本
    if holiday_samples:
        batch_df = pd.concat(holiday_samples, ignore_index=True)
        batch_df = batch_df.drop_duplicates(subset=["kpi_time"])
        holiday_batch_dfs.append(batch_df)
        holiday_samples.clear()
    
    if workday_samples:
        batch_df = pd.concat(workday_samples, ignore_index=True)
        batch_df = batch_df.drop_duplicates(subset=["kpi_time"])
        workday_batch_dfs.append(batch_df)
        workday_samples.clear()
    
    print(f"Generated {len(holiday_batch_dfs)} holiday batches and {len(workday_batch_dfs)} workday batches")
    
    # 最终合并（分批进行）
    print("Final merging and deduplicating...")
    
    # 分批合并holiday数据
    holiday_df = merge_dataframes_in_batches(holiday_batch_dfs)
    workday_df = merge_dataframes_in_batches(workday_batch_dfs)
    
    print(f"Final holiday data: {len(holiday_df)} rows")
    print(f"Final workday data: {len(workday_df)} rows")

    # 对每个模式的分组再次按照时间间隔进行连续性划分
    print("Getting final continuous groups...")
    holiday_groups = get_continuous_group(holiday_df, time_interval) if not holiday_df.empty else []
    workday_groups = get_continuous_group(workday_df, time_interval) if not workday_df.empty else []
    
    print(f"Final holiday groups: {len(holiday_groups)}")
    print(f"Final workday groups: {len(workday_groups)}")
    
    return holiday_groups, workday_groups

def merge_dataframes_in_batches(df_list, batch_size=5):
    """
    分批合并DataFrame列表，避免内存溢出
    """
    if not df_list:
        return pd.DataFrame()
    
    if len(df_list) == 1:
        return df_list[0]
    
    # 分批合并
    while len(df_list) > 1:
        new_df_list = []
        
        for i in tqdm(range(0, len(df_list), batch_size), desc="Merging batches"):
            batch = df_list[i:i + batch_size]
            if batch:
                merged_batch = pd.concat(batch, ignore_index=True)
                merged_batch = merged_batch.drop_duplicates(subset=["kpi_time"])
                new_df_list.append(merged_batch)
                # 释放内存
                del batch
                gc.collect()
        
        df_list = new_df_list
    
    return df_list[0] if df_list else pd.DataFrame()


def save_mode_data(groups_mode, mode,data_basepath="data/0411/ogn/24"):
    """
    保存分组后的数据到指定路径，并按照模式进行分类存储。
    
    :param groups_mode: 分组后的数据列表，每个元素是一个 DataFrame
    :param mode: 模式标识，表示工作日模式 (mode=1) 或周末模式 (mode=0)
    :param his_len: 历史长度，表示每组样本的历史数据长度，默认值为288
    :param pred_len: 预测长度，表示每组样本的预测数据长度，默认值为24
    :param data_basepath: 数据保存的基础路径，默认值为 "data/0411/ogn/24"
    """
    # 构造保存路径，包含历史长度、预测长度和模式信息
    data_basepath = f"{data_basepath}/mode{mode}"
    
    # 如果保存路径不存在，则创建对应的目录
    if not os.path.exists(data_basepath):
        os.makedirs(data_basepath)
    
    print(f"Saving {len(groups_mode)} groups to mode_{mode}...")
    
    # 遍历分组数据列表
    for i, df_temp in enumerate(tqdm(groups_mode, desc=f"Saving mode_{mode} files", unit="file")):
        # 按列 "kpi_time" 去重，确保保存的数据没有重复时间点
        df_temp = df_temp.drop_duplicates(subset=["kpi_time"])
        
        # 将当前分组数据保存为 CSV 文件，文件名为分组索引
        df_temp.to_csv(f"{data_basepath}/{str(i)}.csv", index=False)
    
    print(f"Mode_{mode} data saved successfully!")


def get_continuous_group(df, time_interval):
    """
    根据时间间隔判断连续性分组，根据相邻两条数据是否是连续的时间片来判断
    :param df: 数据框
    :param time_interval: 时间片大小，字符串格式，例如 '5min', '1min', '30s', '1s'
    :return: 连续分组列表
    """
    if df.empty:
        return []
        
    # 转换kpi_time为datetime类型
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    # 对kpi_time进行升序排序
    df = df.sort_values(by="kpi_time")
    
    try:
        time_delta = pd.Timedelta(time_interval)
    except ValueError:
        raise ValueError("Invalid time_interval format. Example valid values: '5min', '1min', '30s', '1s'.")
    
    all_groups = []
    prev_time = None
    current_group = []

    # 遍历数据行
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building continuous groups", unit="row"):
        current_time = row["kpi_time"]
        if prev_time is None or (current_time - prev_time <= time_delta):
            current_group.append(row)
        else:
            if current_group:
                all_groups.append(pd.DataFrame(current_group))
            current_group = [row]
        prev_time = current_time
    
    # 添加最后一个分组
    if current_group:
        all_groups.append(pd.DataFrame(current_group))
    
    return all_groups

if __name__ == "__main__":
    # his_len = 288
    # pred_len = 72

    # data_basepath_list = ["data/pred_6h_args/ogn/24", "data/pred_6h_args/ogn/25"]
    # df_list = [pd.read_csv("data/pred_6h_args/ogn/24/df_2024.csv"), pd.read_csv("data/pred_6h_args/ogn/25/df_2025.csv")]
    # for df, data_basepath in zip(df_list,data_basepath_list):
    #     groups_mode_0, groups_mode_1 = get_group_annotation(his_len=his_len,pred_len=pred_len, df=df)
    #     save_mode_data(
    #         groups_mode=groups_mode_0,
    #         mode=0,
    #         his_len=his_len,
    #         pred_len=pred_len,
    #         data_basepath=data_basepath,
    #     )
    #     save_mode_data(
    #         groups_mode=groups_mode_1,
    #         mode=1,
    #         his_len=his_len,
    #         pred_len=pred_len,
    #         data_basepath=data_basepath,
    #     )
    his_len = 288
    pred_len = 24

    data_basepath_list = ["data/14207/ogn/train"]
    df_list = [pd.read_csv("data/14207/241113_250713.csv")]
    for df, data_basepath in zip(df_list,data_basepath_list):
        groups_mode_0, groups_mode_1 = get_group_annotation(his_len=his_len,pred_len=pred_len, df=df)
        save_mode_data(
            groups_mode=groups_mode_0,
            mode=0,
            his_len=his_len,
            pred_len=pred_len,
            data_basepath=data_basepath,
        )
        save_mode_data(
            groups_mode=groups_mode_1,
            mode=1,
            his_len=his_len,
            pred_len=pred_len,
            data_basepath=data_basepath,
        )


