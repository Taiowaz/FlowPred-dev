import pandas as pd
import numpy as np

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