import pandas as pd
import numpy as np
from datetime import timedelta

def _find_peak_start_time(flow_series: pd.Series, window_size: int) -> pd.Timestamp:
    """辅助函数：找到流量峰值窗口的开始时间"""
    if len(flow_series) < window_size:
        return None
    rolling_sum = flow_series.rolling(window=window_size).sum()
    peak_end_index = rolling_sum.idxmax()
    end_pos = flow_series.index.get_loc(peak_end_index)
    start_pos = end_pos - window_size + 1
    peak_start_time = flow_series.index[start_pos]
    return peak_start_time


def calculate_peak_time_deviation(df: pd.DataFrame, window_duration_min: int = 60) -> float:
    """
    以天为单位，计算预测波峰与实际波峰出现的时间偏差（分钟）。

    Args:
        df: 一个包含三列的Pandas DataFrame:
            - 'time': 预测/真实值对应的时间点（可重复）。
            - 'pred': 模型的预测值。
            - 'real': 对应时间点的真实值。
        window_duration_min: 定义“波峰时段”的窗口宽度，单位为分钟，默认为60。

    Returns:
        一个浮点数，代表预测波峰与真实波峰的中心时间绝对偏差（分钟）。
    """
    if not all(col in df.columns for col in ['time', 'pred', 'real']):
        raise ValueError("DataFrame必须包含 'time', 'pred', 和 'real' 列。")

    # 确保时间列为 datetime 类型
    final_df = (
        df.drop_duplicates(subset='time', keep='last')
        .sort_values('time')
        .assign(time=pd.to_datetime(df['time']))  # 转换时间列为 datetime 类型
        .set_index('time')
    )

    if final_df.empty:
        return np.nan

    time_diffs = final_df.index.to_series().diff().dt.total_seconds() / 60
    time_slice_min = time_diffs.median()

    if pd.isna(time_slice_min) or time_slice_min == 0:
        return np.nan

    window_size = int(round(window_duration_min / time_slice_min))
    if len(final_df) < window_size:
        print("警告: DataFrame中的数据点数量少于一个窗口所需的数据点数量。")
        return np.nan

    actual_peak_start = _find_peak_start_time(final_df['real'], window_size)
    predicted_peak_start = _find_peak_start_time(final_df['pred'], window_size)

    if actual_peak_start is None or predicted_peak_start is None:
        return np.nan

    center_offset = timedelta(minutes=window_duration_min / 2)
    actual_peak_center = actual_peak_start + center_offset
    predicted_peak_center = predicted_peak_start + center_offset

    time_deviation_minutes = abs((predicted_peak_center - actual_peak_center).total_seconds() / 60)

    return time_deviation_minutes