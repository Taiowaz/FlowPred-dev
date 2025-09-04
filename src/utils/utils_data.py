from datetime import datetime
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Process
import holidays
from datetime import time
import pandas as pd


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



def preprocess_for_koopman_30s_moderate(df, spot_id=14207):
    """
    温和版Koopman预处理 - 保持数据真实性
    只处理明显的数据质量问题，避免过度处理
    """
    import numpy as np
    import pandas as pd
    
    print(f"开始温和版Koopman预处理 - 景点{spot_id}")
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
    
    print(f"原始数据统计:")
    print(f"  变异系数: {original_stats['cv']:.3f}")
    print(f"  零值数量: {original_stats['zero_count']}")
    print(f"  数据范围: [{original_stats['min']:.1f}, {original_stats['max']:.1f}]")
    
    processed_values = df['kpi_value'].copy()
    
    # 1. 基础数值清理 - 只处理明显错误
    def clean_basic_issues(series):
        """只清理明显的数值错误"""
        print("  步骤1: 基础数值清理...")
        
        series_clean = series.copy()
        
        # 处理无穷大和NaN
        if np.isinf(series_clean).any() or series_clean.isnull().any():
            # 使用前后值插值，而不是简单的中位数填充
            series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
            series_clean = series_clean.interpolate(method='linear')
            
            # 如果首尾有NaN，用最近的有效值填充
            if series_clean.isnull().any():
                series_clean = series_clean.fillna(method='bfill').fillna(method='ffill')
                # 如果还有NaN，才使用中位数
                if series_clean.isnull().any():
                    series_clean = series_clean.fillna(series_clean.median())
        
        return series_clean
    
    # 2. 温和的连续零值处理 - 提高阈值
    def handle_consecutive_zeros_moderate(series, max_consecutive=60):  # 从15提高到60
        """温和的连续零值处理 - 只处理异常长的零值段"""
        print("  步骤2: 温和连续零值处理...")
        
        series = series.copy()
        zero_mask = (series == 0)
        
        if not zero_mask.any():
            return series
        
        # 找到连续零值的分组
        zero_changes = zero_mask.ne(zero_mask.shift()).cumsum()
        processed_segments = 0
        
        for group_id in zero_changes[zero_mask].unique():
            group_mask = (zero_changes == group_id) & zero_mask
            
            # 只处理异常长的连续零值
            if group_mask.sum() > max_consecutive:
                start_idx = group_mask.idxmax()
                end_idx = group_mask[::-1].idxmax()
                
                # 获取前后非零值
                before_val = None
                after_val = None
                
                # 寻找前面的非零值
                for i in range(start_idx - 1, max(0, start_idx - 50), -1):
                    if series.iloc[i] > 0:
                        before_val = series.iloc[i]
                        break
                
                # 寻找后面的非零值
                for i in range(end_idx + 1, min(len(series), end_idx + 50)):
                    if series.iloc[i] > 0:
                        after_val = series.iloc[i]
                        break
                
                # 如果找不到非零值，使用全局非零均值
                if before_val is None:
                    before_val = series[series > 0].mean() if (series > 0).any() else 10
                if after_val is None:
                    after_val = before_val
                
                # 简单线性插值，不添加复杂的周期性或噪声
                segment_len = group_mask.sum()
                new_values = np.linspace(before_val, after_val, segment_len) * 0.3  # 降低到30%，避免突兀
                
                series.loc[group_mask] = new_values
                processed_segments += 1
        
        print(f"    处理了 {processed_segments} 个异常长连续零值段")
        return series
    
    # 3. 温和的异常值处理 - 不强制改变变异系数
    def handle_extreme_outliers_only(series):
        """只处理极端异常值，保持数据的自然变异性"""
        print("  步骤3: 处理极端异常值...")
        
        # 使用更宽松的异常值检测
        Q1 = series.quantile(0.05)  # 使用更极端的分位数
        Q3 = series.quantile(0.95)
        IQR = Q3 - Q1
        
        if IQR > 0:
            # 使用更宽松的倍数
            k = 3.0  # 从1.5提高到3.0
            lower_bound = max(0, Q1 - k * IQR)
            upper_bound = Q3 + k * IQR
        else:
            # 使用标准差方法，但更宽松
            mean_val = series.mean()
            std_val = series.std()
            lower_bound = max(0, mean_val - 4.0 * std_val)  # 从2.0提高到4.0
            upper_bound = mean_val + 4.0 * std_val
        
        # 只处理真正的极端值
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        
        if outlier_count > 0:
            print(f"    发现 {outlier_count} 个极端异常值")
            # 使用clip进行硬截断，但阈值很宽松
            series_clipped = series.clip(lower=lower_bound, upper=upper_bound)
            return series_clipped
        
        return series
    
    # 4. 确保基本的数值稳定性
    def ensure_basic_stability(series):
        """确保基本的数值稳定性，但不过度处理"""
        print("  步骤4: 确保基本稳定性...")
        
        series_stable = series.copy()
        
        # 确保非负值
        if (series_stable < 0).any():
            negative_count = (series_stable < 0).sum()
            print(f"    发现 {negative_count} 个负值，设置为0")
            series_stable = np.maximum(series_stable, 0)
        
        # 只处理完全相同的长连续段（真正的数据错误）
        same_value_segments = 0
        i = 0
        while i < len(series_stable) - 10:  # 只处理10个以上的相同值
            current_val = series_stable.iloc[i]
            same_count = 1
            
            for j in range(i + 1, len(series_stable)):
                if abs(series_stable.iloc[j] - current_val) < 1e-10:  # 完全相同
                    same_count += 1
                else:
                    break
            
            # 只有连续10个以上完全相同的值才处理
            if same_count >= 10:
                # 添加很小的扰动
                for k in range(i + 1, i + same_count):
                    if k < len(series_stable):
                        series_stable.iloc[k] += np.random.uniform(-0.01, 0.01)
                same_value_segments += 1
                i += same_count
            else:
                i += 1
        
        if same_value_segments > 0:
            print(f"    处理了 {same_value_segments} 个长连续相同值段")
        
        return series_stable
    
    # 执行处理步骤
    print("开始逐步处理:")
    
    # 步骤1: 基础清理
    processed_values = clean_basic_issues(processed_values)
    
    # 步骤2: 温和零值处理
    processed_values = handle_consecutive_zeros_moderate(processed_values, max_consecutive=60)
    
    # 步骤3: 极端异常值处理
    processed_values = handle_extreme_outliers_only(processed_values)
    
    # 步骤4: 基本稳定性
    processed_values = ensure_basic_stability(processed_values)
    
    # 最终检查
    processed_values = processed_values.fillna(method='ffill').fillna(method='bfill')
    if processed_values.isnull().any():
        processed_values = processed_values.fillna(processed_values.median())
    
    # 更新DataFrame
    df['kpi_value'] = processed_values
    
    # 最终统计
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
    
    print(f"\n=== 温和处理结果对比 ===")
    print(f"变异系数: {original_stats['cv']:.3f} -> {final_stats['cv']:.3f} (保持自然变异)")
    print(f"零值数量: {original_stats['zero_count']} -> {final_stats['zero_count']}")
    print(f"数据范围: [{original_stats['min']:.1f}, {original_stats['max']:.1f}] -> [{final_stats['min']:.1f}, {final_stats['max']:.1f}]")
    print(f"NaN/Inf数量: {original_stats['nan_count']}/{original_stats['inf_count']} -> {final_stats['nan_count']}/{final_stats['inf_count']}")
    
    print("✅ 温和版预处理完成 - 保持了数据的自然特征")
    
    return df


def get_spot_config(spot_id,his_hour,pred_hour):
    """
    根据景点ID设置历史长度和预测长度
    """
    if spot_id in [14210,14211,14212,14213]:
        freq = "10m"
        his_len = his_hour * 6
        pred_len = pred_hour * 6
    elif spot_id in [14207,14209]:
        freq = "30s"
        his_len = his_hour * 120
        pred_len = pred_hour * 120
    elif spot_id in [14208]:
        freq = "1m"
        his_len = his_hour * 60
        pred_len = pred_hour * 60
    else:
        freq = "5m"
        his_len = his_hour * 12
        pred_len = pred_hour * 12
    return freq, his_len, pred_len