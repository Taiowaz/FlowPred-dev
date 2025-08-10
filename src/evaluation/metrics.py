import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_daily_rmse_from_df(df: pd.DataFrame) -> float:
    """
    以天为单位，从DataFrame计算数值偏差（RMSE）。

    该函数根据评测方案，计算所有预测值与对应真实值之间的RMSE。

    Args:
        df: 一个包含三列的Pandas DataFrame:
            - 'time': 预测/真实值对应的时间点（可重复）。
            - 'pred': 模型的预测值。
            - 'real': 对应时间点的真实值。

    Returns:
        一个浮点数，代表当天的日度RMSE值。值越小，预测效果越好。
    """
    if not all(col in df.columns for col in ['time', 'pred', 'real']):
        raise ValueError("DataFrame必须包含 'time', 'pred', 和 'real' 列。")

    # 提取真实值和所有预测值
    y_true = df['real']
    y_pred = df['pred']

    # 计算RMSE
    if len(y_true) == 0:
        return 0.0  # 或者返回 np.nan，取决于业务需求

    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    return rmse

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



def calculate_daily_stability(df: pd.DataFrame) -> float:
    """
    以天为单位，计算预测稳定性。

    此函数完全遵循评测方案[cite: 3]。它通过以下步骤工作：
    1. 按被预测的时间点('time')对数据进行分组。
    2. 对每个时间点的所有预测值('pred')计算标准差(Standard Deviation)。
       标准差衡量了对该点的多次预测的离散程度。
    3. 计算这些标准差的日度平均值，作为最终的稳定性指标。

    一个低的值表示模型对未来各时间点的预测比较稳定，波动小。

    Args:
        df: Pandas DataFrame，包含'time'和'pred'列。

    Returns:
        一个浮点数，代表当天预测的平均稳定性。
    """
    if not all(col in df.columns for col in ['time', 'pred']):
        raise ValueError("DataFrame必须包含 'time' 和 'pred' 列。")

    if df.empty:
        return np.nan
        
    # 按时间点分组，并计算每组预测值的标准差
    # ddof=0 表示使用N进行除法，与方案中的公式保持一致（总体标准差）
    stability_per_timeslice = df.groupby('time')['pred'].std(ddof=0)
    
    # 计算这些标准差的平均值
    # 对于只有一个预测的时间点，std结果是NaN，在求平均时忽略它们
    daily_average_stability = stability_per_timeslice.mean()
    
    return daily_average_stability


def calculate_average_trend_accuracy(df: pd.DataFrame, pred_len: int) -> float:
    """
    计算一天内所有预测事件的平均短时趋势准确率。

    此函数遵循评测方案，通过皮尔逊相关系数(ρ)评估趋势准确性。
    1.  它将整个DataFrame按照 `pred_len` 切分成多个“单次预测事件”。
    2.  对每一个独立的预测事件，计算其预测值序列('pred')与真实值序列('real')
        之间的皮尔逊相关系数。
    3.  最后，计算当天所有事件的相关系数值的平均值。

    相关系数(ρ)的值域为[-1, 1]。
    - ρ 越接近1，表示预测趋势与实际趋势越一致（同增同减）。
    - ρ > 0.9 被认为是优秀 [cite: 5]。

    Args:
        df: Pandas DataFrame，需包含'time', 'pred', 'real'列。
        pred_len: 单次预测事件包含的时间片数量。例如，若一次预测未来
                           2小时，每5分钟一个点，则 pred_len 为24。

    Returns:
        一个浮点数，代表当天所有预测事件的平均皮尔逊相关系数。
    """
    if not all(col in df.columns for col in ['time', 'pred', 'real']):
        raise ValueError("DataFrame必须包含 'time', 'pred', 和 'real' 列。")

    if pred_len <= 1:
        raise ValueError("pred_len 必须大于1才能计算相关性。")

    correlations = []
    # 以 pred_len 为步长，遍历DataFrame，切分出每次独立的预测事件
    for i in range(0, len(df), pred_len):
        single_prediction_event = df.iloc[i:i + pred_len]

        # 确保切片长度足够
        if len(single_prediction_event) < pred_len:
            print(f"警告: 数据末尾不足一个完整的预测事件，已忽略。剩余数据行数: {len(single_prediction_event)}")
            continue

        # 计算该事件内，预测与真实值的皮尔逊相关系数
        # Series.corr() 会在标准差为0时（即序列为常量）返回NaN
        correlation = single_prediction_event['pred'].corr(single_prediction_event['real'])
        
        # 只有有效的相关系数值才被纳入计算
        if not np.isnan(correlation):
            correlations.append(correlation)

    # 计算所有事件相关系数的平均值
    if not correlations:
        return np.nan  # 如果没有任何可以计算相关性的事件

    average_correlation = np.mean(correlations)
    
    return average_correlation



def calculate_daily_volatility_ratio(df: pd.DataFrame) -> float:
    """
    以日历史客流为单位，计算波动性/平滑度匹配指标（波动比率 VR）。

    此函数遵循评测方案，用于评估日度预测曲线的波动性与真实情况的匹配度。
    1.  通过保留每个时间点的最新预测，构建“日度历史预测曲线”。
    2.  分别计算历史预测曲线的标准差(stdR)和真实客流曲线的标准差(stdA)。
    3.  计算波动比率 VR = stdR / stdA。

    根据评测方案，VR值越接近1越好。
    - VR > 1 表示预测曲线比实际情况波动更大。
    - VR < 1 表示预测曲线比实际情况更平滑。

    Args:
        df: Pandas DataFrame，需包含'time', 'pred', 'real'列。

    Returns:
        一个浮点数，代表波动比率(VR)。如果无法计算则返回np.nan。
    """
    if not all(col in df.columns for col in ['time', 'pred', 'real']):
        raise ValueError("DataFrame必须包含 'time', 'pred', 和 'real' 列。")

    # 1. 构建日度历史预测曲线（选取每个时间点最新的预测）
    final_df = df.drop_duplicates(subset='time', keep='last')

    if len(final_df) < 2:
        # 数据点少于2个，无法计算标准差
        return np.nan

    # 2. 分别计算真实值和预测值的标准差
    # 使用 ddof=1（样本标准差）是更通用的做法，这里与方案保持一致
    std_real = final_df['real'].std(ddof=1)
    std_pred = final_df['pred'].std(ddof=1)
    
    # 3. 计算波动比率 VR
    if std_real == 0:
        # 如果真实值没有波动，而预测值有波动，则比率为无穷大。
        # 如果预测值也没有波动，则比率为1（完美匹配）。
        return np.inf if std_pred > 0 else 1.0

    # 采用 stdR / stdA 以匹配评价标准的逻辑
    volatility_ratio = std_pred / std_real
    
    return volatility_ratio


def evaluate_single_mode(res_path, pred_len=72):
    df = pd.read_csv(res_path)
    numerical_deviation = calculate_daily_rmse_from_df(df)
    peak_time = calculate_peak_time_deviation(df)
    stability = calculate_daily_stability(df)
    volatility = calculate_daily_volatility_ratio(df)
    trend_accuracy = calculate_average_trend_accuracy(df, pred_len)
    # 输出每个指标及其名称以及该指标怎么算是好的
    print(f"数值偏差 (RMSE): {numerical_deviation:.4f} (越小越好)")
    print(f"峰值时间偏差 (分钟): {peak_time:.4f} (越小越好)")
    print(f"稳定性 (平均标准差): {stability:.4f} (越小越好)")
    print(f"波动率 (VR): {volatility:.4f} (越接近1越好)")
    print(f"趋势准确率 (平均相关系数): {trend_accuracy:.4f} (越接近1越好)")


def evaluate_comparison(mode0_path, mode1_path, pred_len=72):
    """比较两种模式的评估结果"""
    
    # 计算两种模式的指标
    df0 = pd.read_csv(mode0_path)
    df1 = pd.read_csv(mode1_path)
    
    # 节假日指标
    rmse_0 = calculate_daily_rmse_from_df(df0)
    peak_0 = calculate_peak_time_deviation(df0)
    stability_0 = calculate_daily_stability(df0)
    volatility_0 = calculate_daily_volatility_ratio(df0)
    trend_0 = calculate_average_trend_accuracy(df0, pred_len)
    
    # 工作日指标
    rmse_1 = calculate_daily_rmse_from_df(df1)
    peak_1 = calculate_peak_time_deviation(df1)
    stability_1 = calculate_daily_stability(df1)
    volatility_1 = calculate_daily_volatility_ratio(df1)
    trend_1 = calculate_average_trend_accuracy(df1, pred_len)
    
    # 仍然返回DataFrame以便后续使用
    comparison_df = pd.DataFrame({
        '指标': ['数值偏差 (RMSE)', '峰值时间偏差 (分钟)', '稳定性 (平均标准差)', '波动率 (VR)', '趋势准确率 (平均相关系数)'],
        '节假日': [rmse_0, peak_0, stability_0, volatility_0, trend_0],
        '工作日': [rmse_1, peak_1, stability_1, volatility_1, trend_1],
        '评价标准': ['越小越好,小于4000为较好', '越小越好，小于20较好', '越小越好，小于2500较好', '越接近1越好,0.95~1.05较好', '越接近1越好,0.85~1.15较好']
    })
    
    return comparison_df