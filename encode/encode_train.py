import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# 获取分组的编码标注数据,返回两个dataframe列表
# mode=0表示工作日，mode=1表示非工作日

def get_group_annotation(spot_id, df, his_len=288, pred_len=24, data_basepath="data/ogn/mode"):
    # 转换kpi_time为datetime类型
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    # 对kpi_time进行升序排序
    df = df.sort_values(by="kpi_time")
    df["mode"] = df["kpi_time"].dt.weekday.apply(lambda x: 1 if x < 5 else 0)
    # 按天进行分组
    daily_groups = df.groupby(df["kpi_time"].dt.date)
    # 遍历每个分组，如果前后两个分组的时间差为一天，则将它们合并，否则新键一个分组
    all_groups = []
    # 先添加第一个分组

    prev_group_name, first_group = next(iter(daily_groups))
    all_groups.append(first_group)

    # 从第二个分组开始遍历
    for i, (group_name, group) in enumerate(daily_groups):
        if i > 0:
            if (group_name - prev_group_name).days == 1:
                all_groups[-1] = pd.concat([all_groups[-1], group])
            else:
                all_groups.append(group)
            prev_group_name = group_name
    # 遍历每个分组，从每个分组的第his_len个元素开始
    groups_mode_0 = []
    groups_mode_1 = []
    
    # 添加进度条
    for group in tqdm(all_groups, desc=f"{spot_id} Processing groups"):
        if len(group) < his_len + pred_len:
            continue
        # 第一组样例
        # 获取mode第his_len到his_len+pred_len个元素
        data_temp = group[: his_len + pred_len]
        modes = group["mode"][his_len : his_len + pred_len]
        # 统计一下，看看mode=0与mode=1的个数
        num_mode_0 = np.sum(modes == 0)
        num_mode_1 = pred_len - num_mode_0
        if num_mode_1 >= num_mode_0:
            mode = 1
        else:
            mode = 0
        mode_pre = mode
        # 遍历每个分组，从每个分组的第his_len个元素开始
        for i in range(his_len, len(group) - pred_len + 1):
            # 获取第i个元素
            data_i = group.iloc[i]
            # 获取第i+pred_len个元素
            data_j = group.iloc[i + pred_len - 1]
            if data_i["mode"] != data_j["mode"]:
                if data_i["mode"] == 1:
                    num_mode_1 -= 1
                    num_mode_0 += 1
                else:
                    num_mode_0 -= 1
                    num_mode_1 += 1
            # 获取训练集组合
            if mode == mode_pre:
                data_temp = pd.concat([data_temp, data_j.to_frame().T])
                if i == len(group) - pred_len:
                    if mode_pre == 0:
                        groups_mode_0.append(data_temp)
                    else:
                        groups_mode_1.append(data_temp)
            else:
                if mode_pre == 0:
                    groups_mode_0.append(data_temp)
                else:
                    groups_mode_1.append(data_temp)
                # 前his_len到当前元素
                data_temp = group[i - his_len : i]
            mode_pre = mode
    # 保存每个分组
    save_mode_data(spot_id, groups_mode_0, mode=0, his_len=his_len, pred_len=pred_len,data_basepath=data_basepath)
    save_mode_data(spot_id, groups_mode_1, mode=1, his_len=his_len, pred_len=pred_len,data_basepath=data_basepath)
    return groups_mode_0, groups_mode_1



# 保存每种模式的数据
def save_mode_data(spot_id, groups_mode, mode, his_len,pred_len, data_basepath="data/ogn/mode"):
    data_basepath = f"{data_basepath}/{spot_id}/{str(his_len)}_{str(pred_len)}/mode_{mode}"
    if not os.path.exists(data_basepath):
        os.makedirs(data_basepath)
    for i, df_temp in enumerate(groups_mode):
        # 按kpi_time去重
        df_temp = df_temp.drop_duplicates(subset=["kpi_time"])
        df_temp.to_csv(f"{data_basepath}/{str(i)}.csv", index=False)

