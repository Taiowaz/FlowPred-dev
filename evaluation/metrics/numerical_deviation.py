import pandas as pd
import numpy as np

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