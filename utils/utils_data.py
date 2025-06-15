from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process
import holidays
from datetime import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


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


