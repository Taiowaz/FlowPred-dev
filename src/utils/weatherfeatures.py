import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


class WeatherFeatureExtractor:
    """天气特征提取器 - 针对预报数据格式"""
    
    def __init__(self, weather_features: List[str] = None):
        """
        初始化天气特征提取器
        
        Args:
            weather_features: 需要使用的天气特征列表
        """
        # 根据您的数据格式调整特征列表
        self.weather_features = weather_features or [
            'rain', 'weather', 'wind_level', 'wind_direc', 'temp_max', 'temp_min'
        ]
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建衍生特征"""
        df = df.copy()
        
        # 温度相关特征
        df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['temp_comfort'] = ((df['temp_avg'] >= 18) & (df['temp_avg'] <= 26)).astype(float)
        
        # 降雨特征
        df['is_rainy'] = (df['rain'] > 0).astype(float)
        df['rain_level'] = pd.cut(df['rain'], bins=[-0.1, 0, 1, 5, float('inf')], 
                                 labels=[0, 1, 2, 3]).astype(float)
        
        # 风力特征
        df['is_windy'] = (df['wind_level'] >= 3).astype(float)
        df['wind_strength'] = df['wind_level'] / 12.0  # 归一化到0-1
        
        # 天气严重程度（综合指标）
        df['weather_severity'] = (
            (df['rain'] / 10.0).clip(0, 1) +  # 降雨强度
            (df['wind_level'] / 12.0) +       # 风力强度
            (abs(df['temp_avg'] - 20) / 20.0).clip(0, 1)  # 温度偏离舒适区间
        ) / 3.0
        
        return df
    
    def fit(self, weather_df: pd.DataFrame) -> 'WeatherFeatureExtractor':
        """
        拟合天气特征处理器
        
        Args:
            weather_df: 天气数据DataFrame
            
        Returns:
            self
        """
        # 创建衍生特征
        weather_enhanced = self._create_derived_features(weather_df)
        
        # 处理分类特征
        categorical_features = ['weather', 'wind_direc']
        for feature in categorical_features:
            if feature in weather_enhanced.columns:
                le = LabelEncoder()
                le.fit(weather_enhanced[feature].astype(str))
                self.label_encoders[feature] = le
        
        # 确定数值特征
        all_features = self.weather_features + [
            'temp_avg', 'temp_range', 'temp_comfort', 'is_rainy', 'rain_level',
            'is_windy', 'wind_strength', 'weather_severity'
        ]
        numeric_features = [f for f in all_features 
                          if f not in categorical_features and f in weather_enhanced.columns]
        
        # 标准化数值特征
        if numeric_features:
            numeric_data = weather_enhanced[numeric_features].fillna(0)
            self.scaler.fit(numeric_data)
            self.numeric_features = numeric_features
        else:
            self.numeric_features = []
        
        self.is_fitted = True
        return self
    
    def transform(self, weather_df: pd.DataFrame) -> np.ndarray:
        """
        转换天气数据为特征向量
        
        Args:
            weather_df: 天气数据DataFrame
            
        Returns:
            转换后的特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("必须先调用fit方法")
        
        # 创建衍生特征
        weather_enhanced = self._create_derived_features(weather_df)
        
        features = []
        
        # 处理数值特征
        if self.numeric_features:
            numeric_data = weather_enhanced[self.numeric_features].fillna(0)
            scaled_numeric = self.scaler.transform(numeric_data)
            features.append(scaled_numeric)
        
        # 处理分类特征
        for feature, encoder in self.label_encoders.items():
            if feature in weather_enhanced.columns:
                encoded = encoder.transform(weather_enhanced[feature].astype(str))
                features.append(encoded.reshape(-1, 1))
        
        if features:
            return np.concatenate(features, axis=1)
        else:
            return np.array([]).reshape(len(weather_df), 0)
    
    def fit_transform(self, weather_df: pd.DataFrame) -> np.ndarray:
        """拟合并转换数据"""
        return self.fit(weather_df).transform(weather_df)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        if hasattr(self, 'numeric_features'):
            names.extend(self.numeric_features)
        for feature in self.label_encoders.keys():
            names.append(f"{feature}_encoded")
        return names


def preprocess_weather_forecast_data(
    weather_df: pd.DataFrame,
    use_latest_forecast: bool = True
) -> pd.DataFrame:
    """
    预处理天气预报数据，转换为时间序列格式
    
    Args:
        weather_df: 原始天气预报数据
        use_latest_forecast: 是否使用最新的预报数据
        
    Returns:
        处理后的天气数据
    """
    weather_df = weather_df.copy()
    
    # 转换时间列
    weather_df['pred_time'] = pd.to_datetime(weather_df['pred_time'])
    weather_df['start_time'] = pd.to_datetime(weather_df['start_time'])
    weather_df['end_time'] = pd.to_datetime(weather_df['end_time'])
    
    # 如果使用最新预报，对每个时间段选择最新的预报
    if use_latest_forecast:
        weather_df = weather_df.sort_values(['start_time', 'pred_time'])
        weather_df = weather_df.groupby(['start_time', 'end_time']).last().reset_index()
    
    # 创建时间点序列
    processed_data = []
    
    for _, row in weather_df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        
        # 为时间段内的每个时间点创建记录
        current_time = start_time
        while current_time < end_time:
            new_row = row.copy()
            new_row['datetime'] = current_time
            processed_data.append(new_row)
            current_time += timedelta(hours=1)  # 每小时一个数据点
    
    result_df = pd.DataFrame(processed_data)
    result_df = result_df.sort_values('datetime').reset_index(drop=True)
    
    return result_df


def align_weather_with_timeseries(
    weather_df: pd.DataFrame,
    timeseries_dates: pd.Series,
    weather_date_col: str = 'datetime'
) -> pd.DataFrame:
    """
    将天气数据与时间序列数据对齐
    
    Args:
        weather_df: 预处理后的天气数据
        timeseries_dates: 时间序列的日期
        weather_date_col: 天气数据中的日期列名
        
    Returns:
        对齐后的天气数据
    """
    weather_df = weather_df.copy()
    
    # 确保时间列是datetime类型
    weather_df[weather_date_col] = pd.to_datetime(weather_df[weather_date_col])
    timeseries_dates = pd.to_datetime(timeseries_dates)
    
    # 创建时间序列的完整时间范围
    start_time = timeseries_dates.min()
    end_time = timeseries_dates.max()
    
    # 过滤天气数据到相同时间范围
    weather_filtered = weather_df[
        (weather_df[weather_date_col] >= start_time) &
        (weather_df[weather_date_col] <= end_time)
    ].copy()
    
    # 设置datetime为索引并重新索引到时间序列的时间点
    weather_filtered = weather_filtered.set_index(weather_date_col)
    
    # 重新索引到时间序列的时间点
    weather_aligned = weather_filtered.reindex(timeseries_dates, method='ffill')
    
    # 如果还有缺失值，用后向填充
    weather_aligned = weather_aligned.fillna(method='bfill')
    
    # 重置索引
    weather_aligned = weather_aligned.reset_index()
    weather_aligned.rename(columns={'index': weather_date_col}, inplace=True)
    
    return weather_aligned


def expand_weather_to_target_frequency(
    weather_df: pd.DataFrame,
    target_freq: str = '5min',
    weather_date_col: str = 'datetime'
) -> pd.DataFrame:
    """
    将天气数据扩展到目标频率
    
    Args:
        weather_df: 天气数据DataFrame
        target_freq: 目标频率 ('5min', '15min', '1H' 等)
        weather_date_col: 日期列名
        
    Returns:
        扩展后的天气数据
    """
    weather_df = weather_df.copy()
    weather_df[weather_date_col] = pd.to_datetime(weather_df[weather_date_col])
    
    # 创建目标频率的时间范围
    start_time = weather_df[weather_date_col].min()
    end_time = weather_df[weather_date_col].max()
    target_times = pd.date_range(start=start_time, end=end_time, freq=target_freq)
    
    # 设置索引并重新采样
    weather_indexed = weather_df.set_index(weather_date_col)
    weather_resampled = weather_indexed.reindex(target_times, method='ffill')
    
    # 重置索引
    weather_resampled = weather_resampled.reset_index()
    weather_resampled.rename(columns={'index': weather_date_col}, inplace=True)
    
    return weather_resampled


def process_weather_data(
    weather_path: str,
    timeseries_dates: pd.Series,
    weather_features: List[str] = None,
    freq: str = '5min',
    use_latest_forecast: bool = True
) -> Tuple[np.ndarray, WeatherFeatureExtractor]:
    """
    完整的天气数据处理流程
    
    Args:
        weather_path: 天气数据文件路径
        timeseries_dates: 时间序列日期
        weather_features: 使用的天气特征列表
        freq: 目标频率
        use_latest_forecast: 是否使用最新预报
        
    Returns:
        处理后的天气特征矩阵和特征提取器
    """
    try:
        # 读取天气数据
        weather_df = pd.read_csv(weather_path)
        
        # 预处理预报数据
        weather_processed = preprocess_weather_forecast_data(
            weather_df, use_latest_forecast=use_latest_forecast
        )
        
        # 与时间序列对齐
        weather_aligned = align_weather_with_timeseries(
            weather_processed, timeseries_dates
        )
        
        # 扩展到目标频率
        weather_expanded = expand_weather_to_target_frequency(
            weather_aligned, freq
        )
        
        # 特征提取
        extractor = WeatherFeatureExtractor(weather_features)
        weather_features_matrix = extractor.fit_transform(weather_expanded)
        
        return weather_features_matrix, extractor
        
    except Exception as e:
        print(f"处理天气数据时出错: {e}")
        # 返回空数据
        empty_matrix = np.array([]).reshape(len(timeseries_dates), 0)
        empty_extractor = WeatherFeatureExtractor(weather_features)
        return empty_matrix, empty_extractor


def create_weather_encoding_like_timeenc(
    weather_matrix: np.ndarray,
    combine_method: str = 'concat'
) -> np.ndarray:
    """
    将天气特征矩阵转换为类似时间编码的格式
    
    Args:
        weather_matrix: 天气特征矩阵
        combine_method: 组合方法 ('concat', 'mean', 'weighted')
        
    Returns:
        类似时间编码格式的天气特征
    """
    if weather_matrix.size == 0:
        return weather_matrix
    
    if combine_method == 'concat':
        # 直接使用所有特征
        return weather_matrix
    elif combine_method == 'mean':
        # 对特征取平均（降维）
        return np.mean(weather_matrix, axis=1, keepdims=True)
    elif combine_method == 'weighted':
        # 加权组合（可以根据特征重要性调整权重）
        weights = np.array([0.2, 0.15, 0.1, 0.1, 0.15, 0.15, 0.05, 0.05, 0.05])
        if weather_matrix.shape[1] >= len(weights):
            weighted = np.average(weather_matrix[:, :len(weights)], 
                                weights=weights, axis=1)
            return weighted.reshape(-1, 1)
        else:
            return np.mean(weather_matrix, axis=1, keepdims=True)
    else:
        return weather_matrix