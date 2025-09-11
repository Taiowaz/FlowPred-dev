import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ..utils.timefeatures import time_features
from ..utils.weatherfeatures import process_weather_data, create_weather_encoding_like_timeenc  # 新增导入
import warnings

warnings.filterwarnings("ignore")


class Dataset_flow(Dataset):
    def __init__(
        self,
        root_path="data/0411/ogn/25/288/mode_0",
        flag="test",
        size=None,
        features="S",
        data_path=None,
        target="kpi_value",
        scale=False,
        timeenc=0,
        freq="5min",
        datadir_flag=True,
        weather_path=None,  # 新增天气数据路径
        use_weather=False,  # 是否使用天气数据
        weather_encoding_method="concat",  # 新增天气编码方法
        weather_feature_dim=12,  # 新增天气特征维度
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 288
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        if flag == "train" or flag == "val":
            type_map = {"train": 0, "val": 1}
            self.set_type = type_map[flag]
        elif flag == "test":
            self.set_type = 2  # 这里的 0 只是占位，测试时用全量数据

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # 天气数据相关参数
        self.weather_path = weather_path
        self.use_weather = use_weather
        self.weather_encoding_method = weather_encoding_method
        self.weather_feature_dim = weather_feature_dim
        self.weather_extractor = None

        self.root_path = root_path
        self.data_path = data_path
        if datadir_flag:
            self.__read_all_data__()
        else:
            self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # "kpi_time"改为"date"
        df_raw.rename(columns={"kpi_time": "date"}, inplace=True)

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        # 将日期列转换为datetime类型
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        # 按日期排序
        df_raw = df_raw.sort_values(by="date")

        # 检查数据量是否足够
        if len(df_raw) < self.seq_len + self.pred_len:
            print(f"数据量不足，跳过文件: {self.data_path}")
            self.data_x = np.array([])
            self.data_y = np.array([])
            self.data_stamp = np.array([])
            return

        # 找出数据中的间隔
        df_raw["date_diff"] = df_raw["date"].diff().dt.days
        breaks = df_raw[df_raw["date_diff"] > 1].index.tolist()

        # 分割数据
        segments = []
        start_idx = 0
        for break_idx in breaks:
            segments.append(df_raw[start_idx:break_idx])
            start_idx = break_idx
        segments.append(df_raw[start_idx:])

        all_data_x = []
        all_data_y = []
        all_data_stamp = []

        for segment in segments:
            # 检查每个分段的数据量是否足够
            if len(segment) < self.seq_len + self.pred_len:
                continue

            if self.set_type == 0 or self.set_type == 1:  # train 或 val 模式
                num_train = int(len(segment) * 0.8)  # 80% 训练集，20% 验证集
                num_vali = len(segment) - num_train
                border1s = [0, num_train - self.seq_len]
                border2s = [num_train, len(segment)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
            else:  # test 模式，使用全量数据
                border1 = 0
                border2 = len(segment)

            if self.features == "M" or self.features == "MS":
                cols_data = segment.columns[1:]
                df_data = segment[cols_data]
            elif self.features == "S":
                df_data = segment[[self.target]]

            if self.scale:
                train_data = df_data[0:num_train] if self.set_type != 2 else df_data
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = segment[["date"]][border1:border2]
            if self.timeenc == 0:
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
                data_stamp = df_stamp.drop(["date"], axis=1).values
            elif self.timeenc == 1:
                data_stamp = time_features(
                    pd.to_datetime(df_stamp["date"].values), freq=self.freq
                )
                data_stamp = data_stamp.transpose(1, 0)

            # 处理天气数据
            if self.use_weather and self.weather_path:
                try:
                    weather_matrix, self.weather_extractor = process_weather_data(
                        self.weather_path,
                        segment['date'][border1:border2],
                        freq=self.freq
                    )
                    
                    # 确保天气数据长度与时间序列匹配
                    if len(weather_matrix) == (border2 - border1):
                        # 转换为类似时间编码的格式
                        weather_encoding = create_weather_encoding_like_timeenc(
                            weather_matrix, self.weather_encoding_method
                        )
                        # 将天气编码与时间编码合并
                        data_stamp = np.concatenate([data_stamp, weather_encoding], axis=1)
                    else:
                        print(f"天气数据长度不匹配: {len(weather_matrix)} vs {border2 - border1}")
                        
                except Exception as e:
                    print(f"处理天气数据时出错: {e}")

            all_data_x.append(data[border1:border2])
            all_data_y.append(data[border1:border2])
            all_data_stamp.append(data_stamp)

        # 合并所有分段的数据
        self.data_x = np.concatenate(all_data_x, axis=0) if all_data_x else np.array([])
        self.data_y = np.concatenate(all_data_y, axis=0) if all_data_y else np.array([])
        self.data_stamp = (
            np.concatenate(all_data_stamp, axis=0) if all_data_stamp else np.array([])
        )

    def __read_all_data__(self):
        all_data_x = []
        all_data_y = []
        all_data_stamp = []
        # 文件名排序
        files = os.listdir(self.root_path)
        files.sort(key=lambda x: x.split(".")[0])
        for file in files:
            if file.endswith(".csv"):
                self.data_path = file
                self.__read_data__()
                if len(self.data_x) > 0:
                    all_data_x.append(self.data_x)
                    all_data_y.append(self.data_y)
                    all_data_stamp.append(self.data_stamp)
        # 跳过直接拼接，将数据存储为列表
        self.data_x = all_data_x if all_data_x else []
        self.data_y = all_data_y if all_data_y else []
        self.data_stamp = all_data_stamp if all_data_stamp else []

    # __getitem__, __len__, inverse_transform 方法保持不变
    def __getitem__(self, index):
        # 找到合适的分段和分段内的索引
        current_idx = index
        for i, data_segment in enumerate(self.data_x):
            if current_idx < len(data_segment) - self.seq_len - self.pred_len + 1:
                s_begin = current_idx
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = data_segment[s_begin:s_end]
                seq_y = self.data_y[i][r_begin:r_end]
                seq_x_mark = self.data_stamp[i][s_begin:s_end]
                seq_y_mark = self.data_stamp[i][r_begin:r_end]

                return seq_x, seq_y, seq_x_mark, seq_y_mark
            else:
                current_idx -= len(data_segment) - self.seq_len - self.pred_len + 1

        raise IndexError("Index out of range")

    def __len__(self):
        total_len = 0
        for data_segment in self.data_x:
            total_len += len(data_segment) - self.seq_len - self.pred_len + 1
        return total_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Dataset_Pred 类保持不变
class Dataset_Pred(Dataset):
    def __init__(
        self,
        df_raw,
        flag="pred",
        size=None,
        features="S",
        target="kpi_value",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="5min",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 288
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["pred"]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.__read_data__(df_raw)

    def __read_data__(self,df_raw):
        # 将“kpi_time”列名 重命名为 “data”
        df_raw = df_raw.rename(columns={"kpi_time": "date"})
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values(by="date")
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin : r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin : r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

