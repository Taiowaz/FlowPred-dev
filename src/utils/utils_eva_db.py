import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import pymysql
import csv
import torch
import torch.nn as nn



def interactive_plot_comparison(group_labels, loss_list, eva_test_dir, spot_id, *grouped_datasets, save_all=False):
    """
    创建一个交互式绘图，用于比较多个数据集中的真实值和预测值。
    同时在后台自动保存所有图片到指定文件夹。

    参数:
        group_labels (list): 一个标签列表，对应每个分组数据集的标签。
        loss_list (list): 损失值列表。
        eva_test_dir (str): 评估测试目录。
        spot_id (int): 景点ID。
        *grouped_datasets (tuple): 可变数量的分组数据集，每个数据集包含日期和数据。
        save_all (bool): 是否在后台自动保存所有图片，默认False。

    异常:
        ValueError: 如果 group_labels 的长度与 grouped_datasets 的数量不匹配。
    """
    import os
    import threading

    # 检查 group_labels 的长度是否与 grouped_datasets 的数量一致
    if len(group_labels) != len(grouped_datasets):
        raise ValueError("group_labels 的长度必须与 grouped_datasets 的数量匹配。")

    # 当前显示的索引
    current_index = 0

    # 创建保存目录
    save_dir = os.path.join(eva_test_dir, f"{spot_id}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 如果设置了save_all，在后台保存所有图片
    if save_all:
        def background_save():
            for index in range(len(grouped_datasets[0])):
                # 创建一个新的图形，设置图形大小
                plt.figure(figsize=(30, 8))
                
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
                plt.title(f"Spot {spot_id} - {date}")

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

                # 保存图片
                filename = f"spot_{spot_id}_{date}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()  # 关闭图形以节省内存

            
        # 启动后台保存线程
        save_thread = threading.Thread(target=background_save, daemon=True)
        save_thread.start()

    # 创建按钮和输入框
    prev_button = widgets.Button(description="上一个")
    next_button = widgets.Button(description="下一个")
    date_input = widgets.Text(placeholder="输入日期 (YYYY-MM-DD)")
    go_button = widgets.Button(description="跳转")

    # 水平显示按钮和输入框
    button_box = widgets.HBox([prev_button, next_button, date_input, go_button])

    def plot_comparison(index, save_fig=False, show_plot=True):
        """
        绘制指定索引处的真实值和预测值的比较图表。

        参数:
            index (int): 要绘制的数据的索引。
            save_fig (bool): 是否保存图片。
            show_plot (bool): 是否显示图片。
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
        plt.title(f"Spot {spot_id} - {date}")  # 添加景点ID到标题

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

        # 保存图片
        if save_fig:
            filename = f"spot_{spot_id}_{date}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if not show_plot:  # 只在批量保存时打印进度
                print(f"已保存: {filename}")

        if show_plot:
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
        else:
            plt.close()  # 批量保存时关闭图形以节省内存

    def prev_plot(b):
        """处理上一个按钮的点击事件。"""
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            plot_comparison(current_index)

    def next_plot(b):
        """处理下一个按钮的点击事件。"""
        nonlocal current_index
        if current_index < len(grouped_datasets[0]) - 1:
            current_index += 1
            plot_comparison(current_index)

    def go_to_date(b):
        """处理跳转按钮的点击事件，跳转到指定日期。"""
        nonlocal current_index
        date_str = date_input.value
        try:
            target_date = pd.to_datetime(date_str).date()
            for index, (date, _) in enumerate(grouped_datasets[0]):
                if date == target_date:
                    current_index = index
                    plot_comparison(current_index)
                    break
            else:
                print(f"未找到 {date_str} 的数据。")
        except ValueError:
            print("日期格式不正确，请使用 YYYY-MM-DD。")
        # 重新显示当前图片
        plot_comparison(current_index)

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

def view_res(res_path, save_plots=False, eva_test_dir="./plots", spot_id="default"):
    """查看单个结果，可选择在后台保存图片"""
    groups = get_groups_single_mode(res_path)
    args = [groups]
    loss_list = get_loss(*args)
    group_labels = ["pred"]
    
    # 使用交互功能，可选择后台保存
    save_dir = interactive_plot_comparison(group_labels, loss_list, eva_test_dir, spot_id, *args, save_all=save_plots)
    return save_dir

def view_res_multi(*res_paths, prediction_selection="last", pred_len=72, save_plots=False, eva_test_dir="./plots", spot_id="default"):
    """查看多个结果合并，可选择在后台保存图片"""
    groups = get_groups_merge_mutli_df_pred_res(*res_paths, prediction_selection=prediction_selection, pred_len=pred_len)
    args = [groups]
    loss_list = get_loss(*args)
    group_labels = ["pred"]
    
    # 使用交互功能，可选择后台保存
    save_dir = interactive_plot_comparison(group_labels, loss_list, eva_test_dir, spot_id, *args, save_all=save_plots)
    return save_dir


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

def save_all_comparison_plots(group_labels, loss_list, eva_test_dir, spot_id, *grouped_datasets):
    """
    自动保存所有比较图表到指定文件夹。
    
    参数:
        group_labels (list): 一个标签列表，对应每个分组数据集的标签。
        loss_list (list): 损失值列表。
        eva_test_dir (str): 评估测试目录。
        spot_id (int): 景点ID。
        *grouped_datasets (tuple): 可变数量的分组数据集，每个数据集包含日期和数据。
    """
    import os
    
    # 创建保存目录
    save_dir = os.path.join(eva_test_dir, f"{spot_id}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"开始批量保存图片到: {save_dir}")
    print(f"共 {len(grouped_datasets[0])} 张图片...")
    
    for index in range(len(grouped_datasets[0])):
        # 创建一个新的图形，设置图形大小
        plt.figure(figsize=(30, 8))
        
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
        plt.title(f"Spot {spot_id} - {date}")

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

        # 保存图片
        filename = f"spot_{spot_id}_{date}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以节省内存
        
        # 显示进度
        if (index + 1) % 10 == 0 or (index + 1) == len(grouped_datasets[0]):
            print(f"已保存 {index + 1}/{len(grouped_datasets[0])} 张图片")
    
    print(f"✅ 所有图片保存完成！保存路径: {save_dir}")
    return save_dir
