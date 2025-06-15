import pandas as pd
import numpy as np

def encode_prediction(df):
    """
    根据输入的 DataFrame 中前 288 条记录的 kpi_time 列的日期判断是工作日还是周末，进行编码。
    0 代表大部分日期是周末，1 代表大部分日期是工作日。

    :param df: 输入的 DataFrame
    :return: 编码结果，0 或 1
    """
    # 确保 DataFrame 至少有 288 条记录
    if len(df) < 288:
        raise ValueError("输入的 DataFrame 记录数少于 288 条")
    
    # 截取前 288 条记录
    subset_df = df.head(288)
    
    # 转换 kpi_time 为 datetime 类型
    subset_df["kpi_time"] = pd.to_datetime(subset_df["kpi_time"])
    
    # 判断每天是工作日（1）还是周末（0）
    subset_df["is_weekday"] = subset_df["kpi_time"].dt.weekday.apply(lambda x: 1 if x < 5 else 0)
    
    # 统计工作日和周末的数量
    weekday_count = np.sum(subset_df["is_weekday"])
    weekend_count = 288 - weekday_count
    
    # 根据数量判断编码
    if weekday_count >= weekend_count:
        return 1
    else:
        return 0
