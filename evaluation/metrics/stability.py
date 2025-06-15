import pandas as pd
import numpy as np

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