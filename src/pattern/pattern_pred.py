import pandas as pd
import numpy as np
import holidays

def pattern_prediction(df, his_len):
    """
    根据输入的 DataFrame 中前 his_len 条记录的 kpi_time 列的日期判断是工作日还是周末，进行编码。
    0 代表大部分日期是节假日，1 代表大部分日期是工作日。
    """
    if len(df) < his_len:
        raise ValueError(f"输入的 DataFrame 记录数少于 {his_len} 条")
    
    # 截取前 his_len 条记录
    df = df.head(his_len).copy()
    
    # 转换 kpi_time 为 datetime 类型
    df["kpi_time"] = pd.to_datetime(df["kpi_time"])
    
    # 创建中国节假日对象
    china_holidays = holidays.China()
    
    # 判断工作日：不是节假日且不是周末
    def is_workday(dt):
        return (dt.date() not in china_holidays) and (dt.weekday() < 5)
    
    # 计算工作日数量
    workday_count = df["kpi_time"].apply(is_workday).sum()
    
    # 返回编码：工作日多则返回1，否则返回0
    return 1 if workday_count >= (his_len - workday_count) else 0
