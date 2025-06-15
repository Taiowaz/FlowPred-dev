import pandas as pd
import numpy as np

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