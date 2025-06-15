import pandas as pd
import numpy as np
import os
import holidays

# 获取分组的编码标注数据,返回两个dataframe列表
# 效率很低，需要优化
def get_group_annotation(df, his_len=288, pred_len=24, time_interval="5min"):
    """
    根据历史长度和预测长度对数据进行分组，并按照工作日和节假日模式进行分类。
    
    :param df: 数据框，包含时间序列数据
    :param his_len: 历史长度，表示每组样本的历史数据长度，默认值为288
    :param pred_len: 预测长度，表示每组样本的预测数据长度，默认值为24
    :param time_interval: 时间片大小，字符串格式，例如 '5min', '1min', '30s', '1s'
    :return: 两种模式的连续分组列表 (groups_mode_holiday_res, groups_mode_workday_res)
    """
    # 获取连续性分组，初步按照时间间隔划分
    all_groups = get_continuous_group(df, time_interval)
    
    # 创建中国节假日对象
    china_holidays = holidays.China()

    # 初始化两个模式的分组列表
    holiday_samples = []  # 节假日模式样本
    workday_samples = []  # 工作日模式样本
    
    # 预计算历史长度和预测长度总和
    total_len = his_len + pred_len
    
    # 遍历每个连续性分组
    for group in all_groups:
        # 如果分组长度不足以包含历史长度和预测长度，跳过该分组
        if len(group) < total_len:
            continue
        
        # 添加一个新的列 "mode"，根据日期判断工作日 (mode=1) 或节假日、周六周日 (mode=0)
        group["mode"] = group["kpi_time"].apply(lambda x: 0 if x.date() in china_holidays or x.weekday() >= 5 else 1)

        
        # 遍历当前分组，按照历史长度和预测长度拆分样本
        for i in range(len(group) - total_len):
            # 提取当前样本，包含历史数据和预测数据
            group_item = group.iloc[i: i + total_len]
            
            # 统计预测数据段中 (倒数 pred_len 个元素) 的 mode 为 0 的个数
            pred_modes = group_item["mode"][-pred_len:]
            num_mode_0 = np.sum(pred_modes == 0)
            num_mode_1 = pred_len - num_mode_0
            
            # 根据 mode 的数量将样本分配到对应的模式分组
            if num_mode_0 > num_mode_1:
                holiday_samples.append(group_item)
            else:
                workday_samples.append(group_item)
    
    # 合并样本并去重
    holiday_df = pd.concat(holiday_samples, ignore_index=True).drop_duplicates(subset=["kpi_time"]) if holiday_samples else pd.DataFrame()
    workday_df = pd.concat(workday_samples, ignore_index=True).drop_duplicates(subset=["kpi_time"]) if workday_samples else pd.DataFrame()

    # 对每个模式的分组再次按照时间间隔进行连续性划分
    holiday_groups = get_continuous_group(holiday_df, time_interval) if not holiday_df.empty else []
    workday_groups = get_continuous_group(workday_df, time_interval) if not workday_df.empty else []
    
    # 返回两种模式的分组结果
    return holiday_groups, workday_groups


def save_mode_data(groups_mode, mode, his_len=288, pred_len=24, data_basepath="data/0411/ogn/24"):
    """
    保存分组后的数据到指定路径，并按照模式进行分类存储。
    
    :param groups_mode: 分组后的数据列表，每个元素是一个 DataFrame
    :param mode: 模式标识，表示工作日模式 (mode=1) 或周末模式 (mode=0)
    :param his_len: 历史长度，表示每组样本的历史数据长度，默认值为288
    :param pred_len: 预测长度，表示每组样本的预测数据长度，默认值为24
    :param data_basepath: 数据保存的基础路径，默认值为 "data/0411/ogn/24"
    """
    # 构造保存路径，包含历史长度、预测长度和模式信息
    data_basepath = f"{data_basepath}/{str(his_len)}_{pred_len}/mode_{mode}"
    
    # 如果保存路径不存在，则创建对应的目录
    if not os.path.exists(data_basepath):
        os.makedirs(data_basepath)
    
    # 遍历分组数据列表
    for i, df_temp in enumerate(groups_mode):
        # 按列 "kpi_time" 去重，确保保存的数据没有重复时间点
        df_temp = df_temp.drop_duplicates(subset=["kpi_time"])
        
        # 将当前分组数据保存为 CSV 文件，文件名为分组索引
        df_temp.to_csv(f"{data_basepath}/{str(i)}.csv", index=False)


def get_continuous_group(df, time_interval):
    """
    根据时间间隔判断连续性分组，根据相邻两条数据是否是连续的时间片来判断
    :param df: 数据框
    :param time_interval: 时间片大小，字符串格式，例如 '5min', '1min', '30s', '1s'
    :return: 连续分组列表
    """
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
    for _, row in df.iterrows():
        current_time = row["kpi_time"]
        if prev_time is None or (current_time - prev_time <= time_delta):
            current_group.append(row)
        else:
            all_groups.append(pd.DataFrame(current_group))
            current_group = [row]
        prev_time = current_time
    
    # 添加最后一个分组
    if current_group:
        all_groups.append(pd.DataFrame(current_group))
    
    return all_groups

if __name__ == "__main__":
    his_len = 288
    pred_len = 72

    data_basepath_list = ["data/pred_6h_args/ogn/24", "data/pred_6h_args/ogn/25"]
    df_list = [pd.read_csv("data/pred_6h_args/ogn/24/df_2024.csv"), pd.read_csv("data/pred_6h_args/ogn/25/df_2025.csv")]
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

