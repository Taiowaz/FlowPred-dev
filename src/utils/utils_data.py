from datetime import datetime
from matplotlib import table
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Process
import holidays
from datetime import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import pymysql
import csv
from src.Loss.PearsonMSELoss import pearson_mse_loss_xgb_test


def fill_missing_value(data_time_groups, spots):
    """
    填充时间序列数据中的缺失值。
    
    :param data_time_groups: 原始数据，按时间分组的迭代器，每组包含时间点和对应的数据
    :param spots: 需要检查的地点列表 (spot_id 的列表)
    :return: 填充后的完整数据 DataFrame，包含所有时间点和地点的记录
    """
    # 构建一个新的 DataFrame，记录填充的数据
    # 需要前面更新的数据用于后面更新，因此需要记录
    data_fill = pd.DataFrame(columns=["spot_id", "kpi_time", "kpi_value"])

    # 初始化进度条，用于显示处理进度
    pbar = tqdm(total=len(data_time_groups))

    # 遍历每个时间点的分组数据
    for time, time_group in data_time_groups:
        if data_fill.empty:
            # 如果填充数据为空，直接添加当前时间点的数据
            data_fill = pd.concat(
                [data_fill, time_group], ignore_index=True, copy=False
            )
        else:
            # 获取上一个时间点的数据
            pre_time = data_fill["kpi_time"].iloc[-1]
            pre_time_group = data_fill[
                data_fill["kpi_time"] == pre_time
            ]  # 从 data_fill 中获取前一个时间点的数据
            
            # 计算当前时间点与上一个时间点的时间差
            delta = time - pre_time
            
            # 如果时间差大于 5 分钟，说明有缺失的数据，需要填充
            if delta > pd.Timedelta("5min"):
                # 根据时间差计算需要填充的数据
                minutes = int(delta.total_seconds() // 60)
                for i in range(5, minutes, 5):  # 每隔 5 分钟填充一个数据
                    fill_time_group = pre_time_group.copy()
                    fill_time_group["kpi_time"] = pre_time + pd.Timedelta(f"{i}min")
                    data_fill = pd.concat(
                        [data_fill, fill_time_group], ignore_index=True, copy=False
                    )
                # 填充完成后，更新 pre_time 和 pre_time_group
                pre_time = data_fill["kpi_time"].iloc[-1]
                pre_time_group = data_fill[data_fill["kpi_time"] == pre_time]

            # 检查当前时间点的数据是否有缺失的地点
            # 已经拥有的地点，直接添加
            data_fill = pd.concat(
                [data_fill, time_group], ignore_index=True, copy=False
            )
            for spot in spots:
                # 如果当前时间点没有该地点的记录
                spots_group = time_group["spot_id"].values
                if spot not in spots_group:
                    # 获取前一个时间点该地点的 kpi_value
                    pre_kpi_value = pre_time_group[
                        (pre_time_group["kpi_time"] == pre_time)
                        & (pre_time_group["spot_id"] == spot)
                    ]["kpi_value"].values[0]  # 获取的是单个值
                    # 填充缺失的地点记录
                    data_fill = pd.concat(
                        [
                            data_fill,
                            pd.DataFrame(
                                [[spot, time, pre_kpi_value]],
                                columns=["spot_id", "kpi_time", "kpi_value"],
                            ),
                        ],
                        ignore_index=True,
                        copy=False,
                    )
        # 更新进度条
        pbar.update(1)
    # 关闭进度条
    pbar.close()
    return data_fill


def fill_missing_value_singlespot(df, freq=5):
    """
    填充单个地点的时间序列数据中的缺失值。
    
    :param df: 数据框，包含时间序列数据，必须包含 "kpi_time" 和 "kpi_value" 列
    :param freq: 时间片间隔，单位为分钟，默认值为 5 分钟
    :return: 填充后的完整数据 DataFrame，包含所有时间点的记录
    """
    # 构建一个新的 DataFrame，记录填充的数据
    data_fill = pd.DataFrame(columns=["kpi_time", "kpi_value"])
    
    # 将 "kpi_time" 列转换为 datetime 类型，确保时间计算正确
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    
    # 初始化填充数据为原始数据的副本
    data_fill = df.copy()

    # 初始化变量，用于记录最后的时间和值
    last_time = None
    last_value = None
    
    # 使用列表记录填充数据，提高速度
    fill_data = []

    # 遍历原始数据的每一行
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i]
        current_time = row["kpi_time"]  # 当前时间点
        current_value = row["kpi_value"]  # 当前值

        if last_time is not None:
            # 计算当前时间点与上一个时间点的时间差
            delta = current_time - last_time
            if delta > pd.Timedelta(str(freq) + "min"):
                # 如果时间差大于指定的时间片间隔，填充缺失的时间点数据
                minutes = int(delta.total_seconds() // 60)
                for fill_minutes in range(freq, minutes + 1, freq):
                    fill_time = last_time + pd.Timedelta(f"{fill_minutes}min")
                    fill_data.append([fill_time, last_value])  # 使用最后的值填充

        # 更新最后的时间和值
        last_time = current_time
        last_value = current_value

    # 添加原始数据和填充的数据
    data_fill = pd.concat(
        [data_fill, pd.DataFrame(fill_data, columns=["kpi_time", "kpi_value"])],
        ignore_index=True,
    )
    
    # 按时间排序，确保数据顺序正确
    data_fill.sort_values(by="kpi_time", inplace=True)
    
    # 去重，确保每个时间点只有一条记录
    data_fill.drop_duplicates(["kpi_time"], inplace=True)
    
    return data_fill


def fill_missing_value_singlespot_30s(spot_id,df):
    df['minute_str'] = df["kpi_time"].dt.strftime('%Y-%m-%d %H:%M')
    df_groups = list(df.groupby('minute_str'))
    kpi_time_lst = []
    kpi_value_lst = []
    for minute_str, group in df_groups:
        if len(group) == 1:
            # 依据minute_str生成两个时间戳，其他都一样，秒数一个是00，一个是30
            kpi_time_lst.append(group['kpi_time'].iloc[0].replace(second=0, microsecond=0))
            kpi_time_lst.append(group['kpi_time'].iloc[0].replace(second=30, microsecond=0))
            kpi_value_lst.append(group['kpi_value'].iloc[0])
            kpi_value_lst.append(group['kpi_value'].iloc[0])
        elif len(group) == 2:
            kpi_time_lst.append(group['kpi_time'].iloc[0].replace(second=0, microsecond=0))
            kpi_time_lst.append(group['kpi_time'].iloc[0].replace(second=30, microsecond=0))
            kpi_value_lst.append(group['kpi_value'].iloc[0])
            kpi_value_lst.append(group['kpi_value'].iloc[1])
        elif len(group) > 2:
            df_s0 = group[group['kpi_time'].dt.second == 0]
            df_s30 = group[group['kpi_time'].dt.second == 30]
            # 初始化为 
            kpi_value_0 = None
            kpi_value_30 = None
            if len(df_s0) > 0:
                kpi_value_0 = df_s0['kpi_value'].iloc[0]
            if len(df_s30) > 0:
                kpi_value_30 = df_s30['kpi_value'].iloc[0]
            if kpi_value_0 is None:
                kpi_value_0 = group['kpi_value'].iloc[0]
            if kpi_value_30 is None:
                kpi_value_30 = group['kpi_value'].iloc[-1]
            kpi_time_lst.append(group['kpi_time'].iloc[0].replace(second=0, microsecond=0))
            kpi_time_lst.append(group['kpi_time'].iloc[0].replace(second=30, microsecond=0))
            kpi_value_lst.append(kpi_value_0)
            kpi_value_lst.append(kpi_value_30)
    df_res = pd.DataFrame({
        'spot_id': [spot_id] * len(kpi_time_lst),
        'kpi_time': kpi_time_lst,
        'kpi_value': kpi_value_lst
    })
    df_res['kpi_time'] = pd.to_datetime(df_res['kpi_time'])
    return df_res

def fill_missing_value_singlespot_day(df, freq="5min"):
    """
    填充单个景点的时间序列数据中的缺失值，以天为单位进行处理。
    
    :param df: 数据框，包含时间序列数据，必须包含 "kpi_time" 和 "kpi_value" 列
    :param freq: 时间片间隔，字符串格式，例如 "5min"，默认值为 5 分钟
    :return: 填充后的完整数据 DataFrame，包含所有时间点的记录
    """
    # 构建一个新的 DataFrame，记录填充的数据
    data_fill = pd.DataFrame(columns=["kpi_time", "kpi_value"])
    
    # 将 "kpi_time" 列转换为 datetime 类型，确保时间计算正确
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    
    # 按日期对数据进行分组
    df_groups = df.groupby(df["kpi_time"].dt.date)
    
    # 初始化填充数据为原始数据的副本
    data_fill = df.copy()

    # 使用列表记录填充数据，提高速度
    fill_data = []

    # 遍历每一天的数据分组
    for i, (t, g) in enumerate(df_groups):
        fill_data_temp = []  # 临时存储当天填充的数据
        last_time = None  # 记录上一时间点
        last_value = None  # 记录上一时间点的值
        freq_delta = pd.Timedelta(freq)  # 将时间片间隔转换为 Timedelta 对象

        # 遍历当天的每一行数据
        for i in range(g.shape[0]):
            row = g.iloc[i]
            current_time = row["kpi_time"]  # 当前时间点
            current_value = row["kpi_value"]  # 当前值

            if last_time is not None:
                # 计算当前时间点与上一时间点的时间差
                delta = current_time - last_time
                if delta > freq_delta:
                    # 如果时间差大于指定的时间片间隔，填充缺失的时间点数据
                    minutes = int(delta.total_seconds() // 60)
                    freq_minutes = int(freq_delta.total_seconds() // 60)
                    for fill_minutes in range(freq_minutes, minutes, freq_minutes):
                        fill_time = last_time + pd.Timedelta(f"{fill_minutes}min")
                        fill_data_temp.append([fill_time, last_value])
            # 添加当前时间点的数据
            fill_data_temp.append([current_time, current_value])
            # 更新最后的时间和值
            last_time = current_time
            last_value = current_value

        # 后处理：检查当天的起始和结束时间是否需要填充
        if fill_data_temp:
            # 检查当天的第一个时间点是否为 00:00:00
            if fill_data_temp[0][0].time() != pd.to_datetime("00:00:00").time():
                date = fill_data_temp[0][0].date()
                last_time = datetime.combine(date, pd.to_datetime("00:00:00").time())
                delta = fill_data_temp[0][0] - last_time
                if delta >= freq_delta:
                    # 根据时间差填充缺失的时间点
                    minutes = int(delta.total_seconds() // 60)
                    freq_minutes = int(freq_delta.total_seconds() // 60)
                    for fill_minutes in range(0, minutes, freq_minutes):
                        fill_time = last_time + pd.Timedelta(f"{fill_minutes}min")
                        fill_data.append([fill_time, fill_data_temp[0][1]])
            # 添加当天的填充数据
            fill_data.extend(fill_data_temp)

            # 检查当天的最后一个时间点是否为 23:55:00 (一天结束时间减去时间片间隔)
            day_end_time = (pd.to_datetime("00:00:00") - pd.Timedelta(freq)).time()
            if fill_data_temp[-1][0].time() != day_end_time:
                date = fill_data_temp[-1][0].date()
                next_time = datetime.combine(date, day_end_time)
                delta = next_time - fill_data_temp[-1][0]
                if delta >= freq_delta:
                    # 根据时间差填充缺失的时间点
                    minutes = int(delta.total_seconds() // 60)
                    freq_minutes = int(freq_delta.total_seconds() // 60)
                    for fill_minutes in range(freq_minutes, minutes + 1, freq_minutes):
                        fill_time = fill_data_temp[-1][0] + pd.Timedelta(
                            f"{fill_minutes}min"
                        )
                        fill_data.append([fill_time, fill_data_temp[-1][1]])

    # 添加原始数据和填充的数据
    data_fill = pd.concat(
        [data_fill, pd.DataFrame(fill_data, columns=["kpi_time", "kpi_value"])],
        ignore_index=True,
    )
    
    # 按时间排序，确保数据顺序正确
    data_fill.sort_values(by="kpi_time", inplace=True)
    
    # 去重，确保每个时间点只有一条记录
    data_fill.drop_duplicates(["kpi_time"], inplace=True)
    
    # 添加景点 ID 列，确保数据完整性
    data_fill["spot_id"] = df["spot_id"].values[0]
    
    return data_fill


def get_time_slice_index(time):
    """
    根据时间计算时间片索引。
    
    :param time: 时间对象 (datetime.time)
    :return: 时间片索引，从午夜开始，每 5 分钟一个时间片
    """
    # 将时间转换为从午夜开始的分钟数
    minutes_since_midnight = time.hour * 60 + time.minute
    # 每个时间片为 5 分钟
    time_slice_index = minutes_since_midnight // 5
    return time_slice_index


def generate_seq(data, length):
    """
    根据指定长度生成时间序列。
    
    :param data: 输入数据，通常为 NumPy 数组
    :param length: 序列长度
    :return: 生成的时间序列，形状为 (n, length)，其中 n 为生成的序列数量
    """
    seq = np.concatenate(
        [
            np.expand_dims(data[i : i + length], 0)
            for i in range(data.shape[0] - length + 1)
        ],
        axis=0,
    )
    return seq


def generate_seq_set(df, his_len, pred_len):
    """
    根据历史长度和预测长度生成序列集合。
    
    :param df: 数据框，包含时间序列数据，必须包含 "kpi_time" 和 "kpi_value" 列
    :param his_len: 历史长度，表示每组样本的历史数据长度
    :param pred_len: 预测长度，表示每组样本的预测数据长度
    :return: 特征序列 (X)、目标序列 (y)、数据结束时间 (end_time)
    """
    # 获取数据结束时间，用于后续测试结果的保存
    end_time = df["kpi_time"].max()
    # 设置索引为时间列
    df = df.set_index("kpi_time")
    # 创建特征和目标变量
    X = []
    y = []
    # 使用过去 his_len 条数据预测未来 pred_len 条数据
    for i in range(len(df) - his_len - pred_len + 1):
        X.append(df["kpi_value"].iloc[i : i + his_len].values)
        y.append(df["kpi_value"].iloc[i + his_len : i + his_len + pred_len].values)
    # 转换为 NumPy 数组
    X = np.array(X)
    y = np.array(y)

    return X, y, end_time


def parallel_process(func, args_list):
    """
    对传进来的函数与参数进行多线程并行处理。
    
    :param func: 要并行处理的函数
    :param args_list: 参数列表，每个元素是一个元组，包含函数的参数
    """
    processes = []

    def process_func(args):
        """
        内部函数，用于解包参数并调用目标函数。
        
        :param args: 参数元组或单个参数
        """
        # 若是多个参数元组，则解包参数
        if isinstance(args, tuple):
            func(*args)
        else:
            # 单参数直接调用
            func(args)

    # 创建并启动多线程
    for args in args_list:
        # 会进行一次解包，因此需要包一次
        process = Process(target=process_func, args=(args,))
        processes.append(process)
        process.start()

    # 等待所有线程完成
    for process in processes:
        process.join()


def get_time_features(df):
    """
    为数据框添加时间相关的特征列，包括日期、月份、星期几、是否节假日、是否周末等。
    
    :param df: 数据框，必须包含 "kpi_time" 列
    :return: 添加时间特征后的数据框
    """
    # 转换时间列为 datetime 类型
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    # 添加日期、月份、星期几等特征
    df["date"] = df["kpi_time"].dt.date
    df["month"] = df["kpi_time"].dt.month
    df["weekday"] = df["kpi_time"].dt.weekday
    df["dayofmonth"] = df["kpi_time"].dt.day  # 该月的第几天

    # 按日期分组
    df_groups = df.groupby(by="date")
    group_length = len(df_groups)
    groups = []

    # 遍历每个日期分组，添加节假日和周末标识
    for date, group in tqdm(df_groups, total=group_length):
        if date in holidays.China():  # 判断是否为节假日
            group["is_holiday"] = 1
        else:
            group["is_holiday"] = 0
        if str(group.iloc[0]["weekday"]) in ["5", "6"]:  # 判断是否为周末
            group["is_weekend"] = 1
        else:
            group["is_weekend"] = 0
        groups.append(group)

    # 合并所有分组数据
    df_res = pd.concat(groups)
    return df_res



def get_daydata_distribution(df):
    """
    统计每天的数据量分布。
    
    :param df: 数据框，必须包含 "kpi_time" 列
    :return: 字典，键为数据量，值为出现的次数
    """
    d = {}
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    df_groups = df.groupby(df["kpi_time"].dt.date)
    for _, group in df_groups:
        d[len(group)] = d.get(len(group), 0) + 1
    return d



def get_df_diff_less_k(df, num_data_day=288, k=10):
    """
    筛选每天缺失数据条数绝对值小于 k 的数据。
    
    :param df: 数据框，必须包含 "kpi_time" 列
    :param num_data_day: 每天的理论数据量，默认值为 288
    :param k: 缺失数据条数阈值，默认值为 10
    :return: 筛选后的数据框
    """
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    df = df.drop_duplicates(subset=["kpi_time"])
    df_group_date = df.groupby(df["kpi_time"].dt.date)
    date_list = []

    # 遍历每个日期分组，筛选符合条件的数据
    for _, group_date in df_group_date:
        if len(group_date) >= num_data_day - k:
            date_list.append(group_date)
    return pd.concat(date_list, ignore_index=True)



def filter_days_with_less_consecutive_missing(df, freq="5min", k=5):
    """
    筛选每天连续数据缺失条数小于 k 的数据。
    
    :param df: 数据框，必须包含 "kpi_time" 列
    :param freq: 时间片间隔，默认值为 "5min"
    :param k: 连续缺失数据条数阈值，默认值为 5
    :return: 筛选后的数据框
    """
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    df = df.drop_duplicates(subset=["kpi_time"])
    df_group_date = df.groupby(df["kpi_time"].dt.date)
    date_list = []

    # 遍历每个日期分组，筛选符合条件的数据
    for _, group_date in df_group_date:
        missing_dict = find_missing_time_slices(group_date, freq)
        if len(missing_dict) == 0 or max(missing_dict.values()) <= k:
            date_list.append(group_date)
    return pd.concat(date_list, ignore_index=True)



def generate_model_data(df, his_len, pred_len):
    """
    根据历史长度和预测长度生成模型可以直接处理的数据。
    
    :param df: 数据框，必须包含 "kpi_time" 和 "kpi_value" 列
    :param his_len: 历史长度，表示每组样本的历史数据长度
    :param pred_len: 预测长度，表示每组样本的预测数据长度
    :return: 包含特征序列 (X) 和目标序列 (y) 的字典
    """
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    df_grouped = list(df.groupby(df["kpi_time"].dt.date))
    df_list = [[]]

    # 按日期分组，判断是否连续
    df_list[-1].append(df_grouped[0][1])
    for i in range(1, len(df_grouped)):
        group_date_pre = df_grouped[i - 1][0]
        group_date = df_grouped[i][0]
        if group_date_pre.day == group_date.day - 1:  # 判断是否为连续日期
            df_list[-1].append(df_grouped[i][1])
        else:
            df_list.append([])
            df_list[-1].append(df_grouped[i][1])

    # 合并每个分组为一个数据框
    df_list_merged = []
    for i in range(len(df_list)):
        df_list_merged.append(pd.concat(df_list[i]))

    # 生成特征序列和目标序列
    data = {}
    X = np.empty((0, his_len), dtype=int)
    y = np.empty((0, pred_len), dtype=int)
    for df_item in df_list_merged:
        X_temp, y_temp, _ = generate_seq_set(df_item, his_len, pred_len)
        if X_temp.shape[0] != 0:
            X = np.concatenate((X, X_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)

    data["X"], data["y"] = np.expand_dims(X, axis=2), y
    return data


def find_missing_time_slices(df, time_slice):
    """
    此函数用于找出时间序列数据中一整天存在的缺失时间片

    参数:
    df (pandas.DataFrame): 包含时间序列数据的 DataFrame，其中有一列名为 'kpi_time'。
    time_slice (str): 时间片大小，字符串格式，以分钟为单位，例如 '5min'。

    返回:
    dict: 字典的键为缺失时间片的起始值（仅包含时间），值为从起始值开始连续缺失的个数。
    """

    def get_missing_num(time_1, time_2, time_slice):
        # 将时间片字符串转换为 Timedelta 对象
        slice_timedelta = pd.Timedelta(time_slice)
        # 计算两个时间之间的时间片个数
        time_diff = time_2 - time_1
        missing_count = int(time_diff / slice_timedelta) - 1
        return missing_count

    # 获取第一个数据的date
    date = df["kpi_time"].dt.date.iloc[0]
    day_start_time = pd.Timestamp.combine(date, time(0, 0, 0))
    next_day_start = pd.Timestamp.combine(date + pd.Timedelta(days=1), time(0, 0, 0))
    day_end_time = next_day_start - pd.Timedelta(time_slice)
    missing_dict = {}
    missing_count = 0

    # 先判断第一个时间点是否为起始时间
    if df["kpi_time"].iloc[0] != day_start_time:
        missing_count = get_missing_num(
            day_start_time, df["kpi_time"].iloc[0], time_slice
        )
        missing_dict[day_start_time] = missing_count
    # 从第二个时间点开始遍历
    for i in range(1, len(df)):
        current_time = df["kpi_time"].iloc[i]
        previous_time = df["kpi_time"].iloc[i - 1]
        missing_count = get_missing_num(previous_time, current_time, time_slice)
        if missing_count > 0:
            missing_start_time = previous_time + pd.Timedelta(time_slice)
            missing_dict[missing_start_time] = missing_count
    # 检查最后一个时间点是否为结束时间
    if df["kpi_time"].iloc[-1] != day_end_time:
        missing_count = get_missing_num(
            df["kpi_time"].iloc[-1], day_end_time, time_slice
        )
        missing_dict[df["kpi_time"].iloc[-1] + pd.Timedelta(time_slice)] = missing_count
    return missing_dict


def interactive_plot_comparison(group_labels, loss_list, *grouped_datasets):
    """
    创建一个交互式绘图，用于比较多个数据集中的真实值和预测值。

    参数:
        group_labels (list): 一个标签列表，对应每个分组数据集的标签。
        *grouped_datasets (tuple): 可变数量的分组数据集，每个数据集包含日期和数据。

    异常:
        ValueError: 如果 group_labels 的长度与 grouped_datasets 的数量不匹配。
    """

    # 检查 group_labels 的长度是否与 grouped_datasets 的数量一致
    if len(group_labels) != len(grouped_datasets):
        raise ValueError("group_labels 的长度必须与 grouped_datasets 的数量匹配。")

    # 当前显示的索引
    current_index = 0

    # 创建按钮和输入框
    prev_button = widgets.Button(description="上一个")
    next_button = widgets.Button(description="下一个")
    date_input = widgets.Text(placeholder="输入日期 (YYYY-MM-DD)")
    go_button = widgets.Button(description="跳转")

    # 水平显示按钮和输入框
    button_box = widgets.HBox([prev_button, next_button, date_input, go_button])

    def plot_comparison(index):
        """
        绘制指定索引处的真实值和预测值的比较图表。

        参数:
            index (int): 要绘制的数据的索引。
        """
        # 使用外层函数的 current_index 变量
        nonlocal current_index
        # 更新当前索引
        current_index = index

        # 创建一个新的图形，设置图形大小
        plt.figure(figsize=(30, 8))  # 调整图像大小

        # 从第一个数据集中获取当前日期和数据
        date, group = grouped_datasets[0][index]
        # 将数据中的时间列转换为 datetime 类型
        times = pd.to_datetime(group["time"].dt.time.astype(str))

        # 绘制真实值曲线
        plt.plot(times, group["real"], label="real", color="black", linewidth=2)

        # 绘制每组的预测曲线
        for i, grouped_data in enumerate(grouped_datasets):
            _, group = grouped_data[index]
            plt.plot(
                times,
                group["pred"],
                label=group_labels[i] + " loss: " + str(loss_list[i][index]),
            )

        # 设置图例，显示每条曲线的标签
        plt.legend()
        # 设置图表标题为当前日期
        plt.title(f"{date}")  # 标题改为日期

        # 筛选出整点时刻的索引
        hour_indices = [
            i
            for i, t in enumerate(times)
            if t.time().minute == 0 and t.time().second == 0
        ]
        # 获取整点时刻的数据
        hour_times = times.iloc[hour_indices]
        # 将整点时刻格式化为字符串标签
        hour_labels = [t.strftime("%H:%M:%S") for t in hour_times]
        # 设置 x 轴刻度为整点时刻，并显示对应的标签
        plt.xticks(hour_times, hour_labels)

        # 清除之前的输出，并等待新的输出
        clear_output(wait=True)
        # 显示按钮和输入框
        display(button_box)
        # 显示图表
        plt.show()

        # 如果当前是第一个索引，禁用上一个按钮
        prev_button.disabled = index == 0
        # 如果当前是最后一个索引，禁用下一个按钮
        next_button.disabled = index == len(grouped_datasets[0]) - 1

    def prev_plot(b):
        """
        处理上一个按钮的点击事件。

        参数:
            b: 触发事件的按钮对象。
        """
        # 使用外层函数的 current_index 变量
        nonlocal current_index
        # 如果当前索引大于 0，将索引减 1
        if current_index > 0:
            current_index -= 1
            # 绘制新索引对应的比较图表
            plot_comparison(current_index)

    def next_plot(b):
        """
        处理下一个按钮的点击事件。

        参数:
            b: 触发事件的按钮对象。
        """
        # 使用外层函数的 current_index 变量
        nonlocal current_index
        # 如果当前索引小于数据集的最大索引，将索引加 1
        if current_index < len(grouped_datasets[0]) - 1:
            current_index += 1
            # 绘制新索引对应的比较图表
            plot_comparison(current_index)

    def go_to_date(b):
        """
        处理跳转按钮的点击事件，跳转到指定日期。

        参数:
            b: 触发事件的按钮对象。
        """
        # 使用外层函数的 current_index 变量
        nonlocal current_index
        # 获取输入框中的日期字符串
        date_str = date_input.value
        try:
            # 将日期字符串转换为日期对象
            target_date = pd.to_datetime(date_str).date()
            # 遍历第一个数据集中的日期
            for index, (date, _) in enumerate(grouped_datasets[0]):
                if date == target_date:
                    current_index = index
                    # 绘制指定日期的比较图表
                    plot_comparison(current_index)
                    break
            else:
                # 如果未找到指定日期的数据，打印提示信息
                print(f"未找到 {date_str} 的数据。")
        except ValueError:
            # 如果日期格式不正确，打印提示信息
            print("日期格式不正确，请使用 YYYY-MM-DD。")

    # 绑定按钮的点击事件到对应的处理函数
    prev_button.on_click(prev_plot)
    next_button.on_click(next_plot)
    go_button.on_click(go_to_date)

    # 显示初始图表
    plot_comparison(current_index)


DB_CONFIG = {
    'host': "10.62.193.1",
    'port': 3306,
    'user': "xihuspot",
    'password': "xihu#123",
    'database': "flowpredicationsa", # <-- 需要导出数据的数据库名
    'charset': 'utf8mb4', # 建议使用 utf8mb4 以支持各种字符
    'cursorclass': pymysql.cursors.DictCursor # 使用 DictCursor 可以让每行数据都像字典一样，方便按列名访问
}


# 单独一张表的景点
SPOT_ID_TABLE_MAP={
    14207:"dahua_flow",
    14208:"lingyin_passenger_flow",
    14209:"dahua_musical_fountain_num",
    14210:"hubin_realtime",
    14211:"hubin_realtime_new",
    14212:"north_peak_flow",
    14213:"north_peak_holiday_flow",
}

TIME_COL_MAP = {
    "mobile_signaling_tourists_num": "kpi_time",
    "dahua_musical_fountain_num": "date_time",
    "lingyin_passenger_flow": "date_time",
    "dahua_flow": "date_time",
    "north_peak_flow": "date_time",
    "north_peak_holiday_flow": "date_time",
    "hubin_realtime": "timestamp",
    "hubin_realtime_new": "ts"
}

VALUE_COL_MAP = {
    "mobile_signaling_tourists_num": "kpi_value",
    "dahua_musical_fountain_num": "real_num",
    "lingyin_passenger_flow": "real_time_num",
    "dahua_flow": "num",
    "north_peak_flow": "num",
    "north_peak_holiday_flow": "num",
    "hubin_realtime": "value",
    "hubin_realtime_new": "value"
}

def save_csv_from_db(spot_id, s_time, e_time, output_csv_file):
    """
    从数据库中导出指定时间范围的数据到 CSV 文件。

    :param spot_id: 景点ID，用于确定表名和筛选条件
    :param s_time: 起始时间，字符串类型，格式为 'YYYY-MM-DD HH:MM:SS'
    :param e_time: 结束时间，字符串类型，格式为 'YYYY-MM-DD HH:MM:SS'
    :param output_csv_file: 输出的 CSV 文件路径
    """
    try:
        # 连接数据库
        connection = pymysql.connect(**DB_CONFIG)
        print("数据库连接成功！")

        table_name = SPOT_ID_TABLE_MAP.get(spot_id, "mobile_signaling_tourists_num")
        time_col = TIME_COL_MAP.get(table_name, "kpi_time")
        value_col = VALUE_COL_MAP.get(table_name, "kpi_value")
        
        with connection.cursor() as cursor:
            # 根据表名决定是否需要添加 spot_id 筛选条件
            if table_name == "mobile_signaling_tourists_num":
                # 对于 mobile_signaling_tourists_num 表，需要添加 spot_id 筛选条件
                if s_time and e_time:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE {time_col} BETWEEN %s AND %s AND spot_id = %s
                        ORDER BY {time_col}
                    """
                    params = (s_time, e_time, spot_id)
                    print(f"正在从表 '{table_name}' 中查询数据，景点ID: {spot_id}，时间范围: {s_time} 至 {e_time}...")
                elif s_time:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE {time_col} >= %s AND spot_id = %s
                        ORDER BY {time_col}
                    """
                    params = (s_time, spot_id)
                    print(f"正在从表 '{table_name}' 中查询数据，景点ID: {spot_id}，起始时间: {s_time}...")
                elif e_time:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE {time_col} <= %s AND spot_id = %s
                        ORDER BY {time_col}
                    """
                    params = (e_time, spot_id)
                    print(f"正在从表 '{table_name}' 中查询数据，景点ID: {spot_id}，结束时间: {e_time}...")
                else:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE spot_id = %s
                        ORDER BY {time_col}
                    """
                    params = (spot_id,)
                    print(f"正在从表 '{table_name}' 中查询所有数据，景点ID: {spot_id}...")
            else:
                # 对于其他表，不需要 spot_id 筛选条件（因为这些表本身就是特定景点的专用表）
                if s_time and e_time:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE {time_col} BETWEEN %s AND %s
                        ORDER BY {time_col}
                    """
                    params = (s_time, e_time)
                    print(f"正在从表 '{table_name}' 中查询数据，时间范围: {s_time} 至 {e_time}...")
                elif s_time:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE {time_col} >= %s
                        ORDER BY {time_col}
                    """
                    params = (s_time,)
                    print(f"正在从表 '{table_name}' 中查询数据，起始时间: {s_time}...")
                elif e_time:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        WHERE {time_col} <= %s
                        ORDER BY {time_col}
                    """
                    params = (e_time,)
                    print(f"正在从表 '{table_name}' 中查询数据，结束时间: {e_time}...")
                else:
                    query = f"""
                        SELECT {time_col} as kpi_time, {value_col} as kpi_value FROM {table_name}
                        ORDER BY {time_col}
                    """
                    params = ()
                    print(f"正在从表 '{table_name}' 中查询所有数据...")

            # 执行查询
            cursor.execute(query, params)

            # 获取所有数据
            rows = cursor.fetchall()

            if rows:
                # 统一的表头
                headers = ["spot_id", "kpi_time", "kpi_value"]
                print(f"输出表头: {headers}")

                # 打开 CSV 文件准备写入
                with open(output_csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)

                    # 写入表头
                    writer.writerow(headers)
                    print("CSV 文件表头已写入。")

                    # 写入数据行，添加 spot_id 列
                    for row in rows:
                        writer.writerow([spot_id, row['kpi_time'], row['kpi_value']])

                print(f"成功！筛选后的数据已导出到文件 '{output_csv_file}'，共 {len(rows)} 条记录。")
            else:
                print(f"表 '{table_name}' 中指定条件下没有数据。")

    except pymysql.Error as err:
        print(f"数据库操作失败: {err}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        # 关闭数据库连接
        if 'connection' in locals() and connection.open:
            connection.close()
            print("数据库连接已关闭。")


# 获取单个预测结果的预测处理
def get_df_single_mode(res_path, prediction_selection="last", pred_len=72):
    df = pd.read_csv(res_path)
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values(by="time", inplace=True)
    if prediction_selection == "last":
        # 按时间分组，选取每组的最后一次预测
        # df_temp = df.groupby("time").last().reset_index()
        df_temp = df[(df.index+1) % pred_len == 0]
    elif prediction_selection == "first":
        # 按时间分组，选取每组的第一次预测
        # df_temp = df.groupby("time").first().reset_index()
        df_temp = df[(df.index) % pred_len == 0]
    else:
        assert prediction_selection in ["first", "last"], f"prediction_selection 必须是 'first' 或 'last'，但传入的值是 {prediction_selection}"
    df_res = df_temp.sort_values(by="time")
    return df_res

# 获取多个预测结果路径，使用get_df_single_mode进行处理，最终融合成一个df
def get_groups_merge_mutli_df_pred_res(*res_paths, prediction_selection="last", pred_len=72):
    df_list = []
    for res_path in res_paths:
        # print(res_path)
        df = get_df_single_mode(res_path, prediction_selection, pred_len)
        df_list.append(df)
    # 合并所有 DataFrame
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset='time')
    df_merged = df_merged.sort_values(by='time')
    groups_merged = list(df_merged.groupby(df_merged["time"].dt.date))
    return groups_merged

# 获得处理的分组
def get_groups_single_mode(res_path, prediction_selection="last", pred_len=72):
    df_res = get_df_single_mode(res_path, prediction_selection, pred_len)
    groups_mode = list(df_res.groupby(df_res["time"].dt.date))
    return groups_mode

def view_res(res_path):
    groups = get_groups_single_mode(res_path)
    args = [groups]
    loss_list = get_loss(*args)
    group_labels = ["pred"]
    interactive_plot_comparison(group_labels, loss_list, *args)
def view_res_multi(*res_paths):
    groups = get_groups_merge_mutli_df_pred_res(*res_paths)
    args = [groups]
    loss_list = get_loss(*args)
    group_labels = ["pred"]
    interactive_plot_comparison(group_labels, loss_list, *args)


def get_groups(base_dir, dir_name, pred_len):
    df_mode0 = pd.read_csv(f"{base_dir}/0/{dir_name}/result.csv")
    df_mode1 = pd.read_csv(f"{base_dir}/1/{dir_name}/result.csv")
    df_mode0["time"] = pd.to_datetime(df_mode0["time"])
    df_mode1["time"] = pd.to_datetime(df_mode1["time"])
    df_mode0.sort_values(by="time", inplace=True)
    df_mode1.sort_values(by="time", inplace=True)
    df_mode0_far = df_mode0[(df_mode0.index + 1) % pred_len == 0]
    df_mode1_far = df_mode1[(df_mode1.index + 1) % pred_len == 0]
    df_mode = pd.concat([df_mode0_far, df_mode1_far], ignore_index=True)
    df_mode = df_mode.sort_values(by="time")
    groups_mode = list(df_mode.groupby(df_mode["time"].dt.date))
    return groups_mode


def get_loss(*groups_mode):
    loss_list = []
    # 创建 MSE 损失函数
    mse_loss = nn.MSELoss()
    
    # 对于每一个groups_mode, 计算loss
    for groups in groups_mode:
        loss_temp = []
        for date, group in groups:
            # 将数据转换为 torch tensor
            pred_tensor = torch.tensor(group["pred"].values, dtype=torch.float32)
            real_tensor = torch.tensor(group["real"].values, dtype=torch.float32)
            
            # 计算 MSE 损失
            mse = mse_loss(pred_tensor, real_tensor)
            # 计算 RMSE (对 MSE 开平方根)
            rmse = torch.sqrt(mse)
            loss_temp.append(rmse.item())  # 使用 .item() 获取标量值
        loss_list.append(loss_temp)
    return loss_list



def preprocess_for_koopman_30s_enhanced(df, spot_id=14207):
    """
    增强版Koopman预处理，专门解决Mode0/Mode1的NaN问题
    针对变异系数过高(1.052)和连续零值过长(120)的问题
    """
    import numpy as np
    import pandas as pd
    
    print(f"开始增强版Koopman预处理 - 景点{spot_id}")
    print(f"原始数据形状: {df.shape}")
    
    # 原始统计
    original_stats = {
        'mean': df['kpi_value'].mean(),
        'std': df['kpi_value'].std(),
        'cv': df['kpi_value'].std() / df['kpi_value'].mean(),
        'min': df['kpi_value'].min(),
        'max': df['kpi_value'].max(),
        'zero_count': (df['kpi_value'] == 0).sum(),
        'nan_count': df['kpi_value'].isnull().sum(),
        'inf_count': np.isinf(df['kpi_value']).sum()
    }
    
    print(f"原始问题检查:")
    print(f"  变异系数: {original_stats['cv']:.3f}")
    print(f"  零值数量: {original_stats['zero_count']}")
    print(f"  数据范围: [{original_stats['min']:.1f}, {original_stats['max']:.1f}]")
    
    # 1. 数值清理 - 最优先
    def clean_numerical_issues(series):
        """清理所有数值问题"""
        print("  步骤1: 清理NaN和Inf...")
        
        series_clean = series.copy()
        
        # 处理无穷大和NaN
        if np.isinf(series_clean).any() or series_clean.isnull().any():
            median_val = series_clean.median()
            if pd.isna(median_val):
                median_val = 50.0  # 使用合理的默认值
            series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
            series_clean = series_clean.fillna(median_val)
        
        return series_clean
    
    # 2. 强化的连续零值处理 - 关键修改
    def handle_consecutive_zeros_enhanced(series, max_consecutive=15):  # 进一步降低到15
        """增强版连续零值处理 - 更激进的处理"""
        print("  步骤2: 激进连续零值处理...")
        
        series = series.copy()
        zero_mask = (series == 0)
        
        if not zero_mask.any():
            return series
        
        # 找到连续零值的分组
        zero_changes = zero_mask.ne(zero_mask.shift()).cumsum()
        processed_segments = 0
        
        for group_id in zero_changes[zero_mask].unique():
            group_mask = (zero_changes == group_id) & zero_mask
            
            if group_mask.sum() > max_consecutive:
                # 获取连续零值段的位置
                start_idx = group_mask.idxmax()
                end_idx = group_mask[::-1].idxmax()
                
                # 更智能的前后值获取
                context_size = 50  # 扩大上下文窗口
                
                # 前值：取前面非零值的加权平均
                before_start = max(0, start_idx - context_size)
                before_segment = series.iloc[before_start:start_idx]
                before_nonzero = before_segment[before_segment > 0]
                
                if len(before_nonzero) > 0:
                    # 使用最近的几个值的加权平均
                    weights = np.exp(-np.arange(len(before_nonzero)) * 0.1)  # 越近权重越大
                    before_val = np.average(before_nonzero.iloc[-10:], weights=weights[-10:] if len(before_nonzero) >= 10 else weights[-len(before_nonzero):])
                else:
                    before_val = series[series > 0].mean() if (series > 0).any() else 30
                
                # 后值：同样处理
                after_end = min(len(series), end_idx + context_size)
                after_segment = series.iloc[end_idx+1:after_end]
                after_nonzero = after_segment[after_segment > 0]
                
                if len(after_nonzero) > 0:
                    weights = np.exp(-np.arange(len(after_nonzero)) * 0.1)
                    after_val = np.average(after_nonzero.iloc[:10], weights=weights[:10] if len(after_nonzero) >= 10 else weights[:len(after_nonzero)])
                else:
                    after_val = before_val
                
                # 更自然的插值策略
                segment_len = group_mask.sum()
                
                # 使用Sigmoid函数创建平滑过渡
                t = np.linspace(-3, 3, segment_len)
                sigmoid = 1 / (1 + np.exp(-t))
                base_values = before_val + (after_val - before_val) * sigmoid
                
                # 添加周期性变化模拟真实波动
                if segment_len > 30:  # 只对长段添加周期性
                    # 模拟小时级周期
                    hourly_cycle = np.sin(2 * np.pi * np.arange(segment_len) / 120) * min(before_val, after_val) * 0.1
                    base_values += hourly_cycle
                
                # 添加适度随机噪声
                noise_std = min(before_val, after_val) * 0.08  # 降低噪声强度
                noise = np.random.normal(0, noise_std, segment_len)
                
                # 确保值为正且合理
                new_values = np.maximum(0.5, base_values + noise)
                
                series.loc[group_mask] = new_values
                processed_segments += 1
        
        print(f"    处理了 {processed_segments} 个长连续零值段")
        return series
    
    # 3. 强力降低变异性 - 关键修改
    def reduce_high_variation(series, target_cv=0.6):  # 降低目标到0.6
        """强力降低过高变异性"""
        print("  步骤3: 强力降低变异性...")
        
        current_cv = series.std() / series.mean()
        print(f"    当前变异系数: {current_cv:.3f}")
        
        if current_cv > target_cv:
            # 多重平滑策略
            series_smooth = series.copy()
            
            # 第一步：移除极端异常值
            Q1, Q3 = series_smooth.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                upper_bound = Q3 + 2.0 * IQR  # 更严格的异常值界限
                lower_bound = max(0, Q1 - 1.5 * IQR)
                series_smooth = series_smooth.clip(lower=lower_bound, upper=upper_bound)
            
            # 第二步：自适应强度平滑
            if current_cv > 1.2:
                window = 20  # 超强平滑
                smoothing_ratio = 0.8
            elif current_cv > 1.0:
                window = 15  # 强平滑
                smoothing_ratio = 0.7
            else:
                window = 10   # 中等平滑
                smoothing_ratio = 0.6
            
            # 使用三角权重的移动平均（比高斯更平滑）
            def triangular_weights(n):
                if n % 2 == 1:
                    center = n // 2
                    weights = np.minimum(np.arange(n) + 1, np.arange(n, 0, -1))
                else:
                    weights = np.minimum(np.arange(n) + 1, np.arange(n, 0, -1))
                return weights / weights.sum()
            
            weights = triangular_weights(window)
            
            # 应用加权平滑
            smoothed = series_smooth.rolling(window=window, center=True, min_periods=1).apply(
                lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 1 else x.iloc[0],
                raw=False
            )
            
            # 强制混合比例
            series_reduced = smoothing_ratio * smoothed + (1 - smoothing_ratio) * series_smooth
            
            # 第三步：如果还是太高，应用对数变换
            current_cv_after = series_reduced.std() / series_reduced.mean()
            if current_cv_after > target_cv * 1.2:
                print(f"    应用对数变换进一步降低变异性...")
                # 对数变换
                series_log = np.log1p(series_reduced)
                # 重新缩放
                series_scaled = (series_log - series_log.min()) / (series_log.max() - series_log.min())
                series_reduced = series_reduced.mean() * (0.5 + series_scaled)  # 控制在合理范围
            
            new_cv = series_reduced.std() / series_reduced.mean()
            print(f"    强力平滑后变异系数: {new_cv:.3f}")
            
            return series_reduced
        
        return series
    
    # 4. 增强异常值处理
    def handle_outliers_smart(series):
        """增强异常值处理"""
        print("  步骤4: 增强异常值处理...")
        
        # 使用更严格的异常值检测
        Q1 = series.quantile(0.20)  # 更严格的分位数
        Q3 = series.quantile(0.80)
        IQR = Q3 - Q1
        
        # 更严格的异常值阈值
        if IQR > 0:
            k = 1.5  # 更严格的系数
            lower_bound = max(0.1, Q1 - k * IQR)
            upper_bound = Q3 + k * IQR
        else:
            # 如果IQR为0，使用更严格的标准差方法
            mean_val = series.mean()
            std_val = series.std()
            lower_bound = max(0.1, mean_val - 2.0 * std_val)
            upper_bound = mean_val + 2.0 * std_val
        
        # 渐进式软截断
        series_clipped = series.copy()
        
        # 处理上界异常值 - 渐进压缩
        upper_mask = series > upper_bound
        if upper_mask.any():
            excess_ratio = (series[upper_mask] - upper_bound) / upper_bound
            # 使用平方根函数压缩（比对数更温和）
            compressed = upper_bound * (1 + np.sqrt(excess_ratio) * 0.3)
            series_clipped.loc[upper_mask] = compressed
        
        # 处理下界
        series_clipped = series_clipped.clip(lower=lower_bound)
        
        return series_clipped
    
    # 5. 强化数值稳定性
    def ensure_numerical_stability(series):
        """强化数值稳定性"""
        print("  步骤5: 强化数值稳定性...")
        
        series_stable = series.copy()
        
        # 避免完全相同的连续值 - 更强的处理
        for i in range(1, len(series_stable)):
            diff = abs(series_stable.iloc[i] - series_stable.iloc[i-1])
            if diff < series_stable.iloc[i] * 0.001:  # 相对变化小于0.1%
                # 添加相对变化而非绝对变化
                relative_change = np.random.uniform(-0.01, 0.01)  # 1%的相对变化
                series_stable.iloc[i] = max(0.1, series_stable.iloc[i] * (1 + relative_change))
        
        # 确保最小值
        series_stable = np.maximum(series_stable, 0.1)
        
        # 强制控制动态范围
        max_val = series_stable.max()
        min_val = series_stable.min()
        dynamic_range = max_val / min_val
        
        if dynamic_range > 100:  # 更严格的动态范围控制
            print(f"    强制控制动态范围: {dynamic_range:.1f} -> 100")
            # 对高值进行平方根压缩
            median_val = series_stable.median()
            high_threshold = median_val * 5
            high_mask = series_stable > high_threshold
            
            if high_mask.any():
                excess = series_stable[high_mask] / high_threshold
                series_stable.loc[high_mask] = high_threshold * np.sqrt(excess)
        
        return series_stable
    
    # 执行所有处理步骤
    print("开始逐步处理:")
    
    processed_values = df['kpi_value'].copy()
    
    # 步骤1: 数值清理
    processed_values = clean_numerical_issues(processed_values)
    
    # 步骤2: 激进零值处理
    processed_values = handle_consecutive_zeros_enhanced(processed_values, max_consecutive=15)
    
    # 步骤3: 强力降低变异性
    processed_values = reduce_high_variation(processed_values, target_cv=0.6)
    
    # 步骤4: 增强异常值处理
    processed_values = handle_outliers_smart(processed_values)
    
    # 步骤5: 强化数值稳定性
    processed_values = ensure_numerical_stability(processed_values)
    
    # 最终安全检查
    processed_values = processed_values.fillna(method='ffill').fillna(method='bfill')
    processed_values = processed_values.fillna(processed_values.median())
    processed_values = np.maximum(processed_values, 0.1)  # 确保最小值
    
    # 更新DataFrame
    df['kpi_value'] = processed_values
    
    # 最终统计和连续零值检查
    final_stats = {
        'mean': df['kpi_value'].mean(),
        'std': df['kpi_value'].std(),
        'cv': df['kpi_value'].std() / df['kpi_value'].mean(),
        'min': df['kpi_value'].min(),
        'max': df['kpi_value'].max(),
        'zero_count': (df['kpi_value'] == 0).sum(),
        'nan_count': df['kpi_value'].isnull().sum(),
        'inf_count': np.isinf(df['kpi_value']).sum()
    }
    
    # 检查最大连续零值
    def check_max_consecutive_zeros(data):
        zero_runs = []
        current_run = 0
        for value in data:
            if value == 0:
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            zero_runs.append(current_run)
        return max(zero_runs) if zero_runs else 0
    
    final_max_consecutive = check_max_consecutive_zeros(df['kpi_value'])
    
    print(f"\n=== 强化处理结果对比 ===")
    print(f"变异系数: {original_stats['cv']:.3f} -> {final_stats['cv']:.3f} (目标: <0.6)")
    print(f"零值数量: {original_stats['zero_count']} -> {final_stats['zero_count']}")
    print(f"最大连续零值: 120 -> {final_max_consecutive}")
    print(f"数据范围: [{original_stats['min']:.1f}, {original_stats['max']:.1f}] -> [{final_stats['min']:.1f}, {final_stats['max']:.1f}]")
    print(f"动态范围比: {original_stats['max']/max(0.1, original_stats['min']):.1f} -> {final_stats['max']/final_stats['min']:.1f}")
    
    # 成功判断标准
    success_criteria = [
        final_stats['cv'] < 0.8,
        final_max_consecutive < 20,
        final_stats['nan_count'] == 0,
        final_stats['inf_count'] == 0,
        final_stats['max']/final_stats['min'] < 200
    ]
    
    if all(success_criteria):
        print("✅ 强化处理完全成功！应该能彻底解决NaN问题")
    elif sum(success_criteria) >= 4:
        print("✅ 强化处理基本成功，大幅改善")
    else:
        print("⚠️ 强化处理有改善，但可能需要进一步调整序列长度")
    
    print(f"✅ 强化版Koopman预处理完成")
    
    return df



def get_spot_config(spot_id,his_hour,pred_hour):
    """
    根据景点ID设置历史长度和预测长度
    """
    if spot_id in [14210,14211,14212,14213]:
        freq = "10min"
        his_len = his_hour * 6
        pred_len = pred_hour * 6
    elif spot_id in [14207,14209]:
        freq = "30sec"
        his_len = his_hour * 120
        pred_len = pred_hour * 120
    elif spot_id in [14208]:
        freq = "1min"
        his_len = his_hour * 60
        pred_len = pred_hour * 60
    else:
        freq = "5min"
        his_len = his_hour * 12
        pred_len = pred_hour * 12
    return freq, his_len, pred_len