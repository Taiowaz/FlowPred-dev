from calendar import c
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
warnings.filterwarnings("ignore")

from src.data_provider.data_factory import data_provider
from src.exp.exp_basic import Exp_Basic
from src.models import Koopa
from src.utils.tools import EarlyStopping, adjust_learning_rate, visual
from src.Loss.PearsonMSELoss import PearsonMSELoss



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args.label_len = self.args.pred_hour
        

    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_data, train_loader = self._get_data(flag="train")
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0] * self.args.alpha)).indices
        return mask_spectrum  # as the spectrums of time-invariant component

    def _build_model(self):
        model_dict = {
            "Koopa": Koopa,
        }
        mask_spectrum_dir = os.path.join(
            self.args.mask_spectrum_dir,
            f"his{str(self.args.his_hour)}h_pred{str(self.args.pred_hour)}h", 
            str(self.args.spot_id),# 确保 seq_len 是字符串类型
            f"mode{str(self.args.mode)}",
        )

        if self.args.is_training:
            # 训练阶段，计算并保存 mask_spectrum
            mask_spectrum = self._get_mask_spectrum()
            if not os.path.exists(mask_spectrum_dir):
                os.makedirs(mask_spectrum_dir)
            np.save(f"{mask_spectrum_dir}/mask.npy", mask_spectrum.cpu().numpy())
            self.args.mask_spectrum = mask_spectrum
        else:
            print("mask_spectrum_dir:", mask_spectrum_dir)
            # 预测阶段，加载 mask_spectrum
            if os.path.exists(self.args.mask_spectrum_dir):
                mask_spectrum = np.load(f"{mask_spectrum_dir}/mask.npy")
                self.args.mask_spectrum = torch.from_numpy(mask_spectrum).to(self.device)
            else:
                raise FileNotFoundError(f"Mask spectrum file not found at {self.args.mask_spectrum_load_path}")

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        self.set_root_path()
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = PearsonMSELoss(lambda_pearson=0.4)
        criterion = nn.MSELoss() 
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def get_single_prediction_time(self, test_loader):
        """
        计算单次预测的时间
        :param test_loader: 测试数据加载器
        :return: 单次预测的时间（秒）
        """
        self.model.eval()
        with torch.no_grad():
            # 取第一个测试用例
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )

            start_time = time.time()  # 记录预测开始时间
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            end_time = time.time()  # 记录预测结束时间

        return end_time - start_time

    def train(self):
        # 定义模型保存路径
        model_save_path = os.path.join(
            self.args.checkpoints,
            f"his{str(self.args.his_hour)}h_pred{str(self.args.pred_hour)}h", 
            str(self.args.spot_id),
            f"mode{str(self.args.mode)}",
        )
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        # 记录训练开始时间
        start_time = time.time()

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = model_save_path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        # 记录训练结束时间并计算总时间
        end_time = time.time()
        total_train_time = end_time - start_time
        print(f"训练完成，总共耗时: {total_train_time:.4f} 秒")
        # 记录单次训练时间
        time_single_pred = self.get_single_prediction_time(test_loader=test_loader)
        print(f"单次预测时间: {time_single_pred}")
        return self.model

    # 获取时间数据
    def get_time_df_data_test(self):
        time_stamps = []
        # 文件名排序
        files = os.listdir(self.args.root_path)
        files.sort(key=lambda x: x.split(".")[0])  # 假设文件名是数字，按数字排序
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(self.args.root_path, file)
                df = pd.read_csv(file_path)
                # 滑动窗口，窗口大小为 pred_len, 从seq_len开始，每次滑动1个单位
                for i in range(self.args.seq_len, len(df) - self.args.pred_len + 1):
                    # 将窗口内所有时间戳添加到列表中
                    time_stamps.extend(df["kpi_time"][i : i + self.args.pred_len])
            print(f"{file} time_stamps: ", len(time_stamps))
        return time_stamps

    def test(self):
        test_data, test_loader = self._get_data(flag="test")
        print("loading model")
        model_save_path = os.path.join(
            self.args.checkpoints,
            f"his{str(self.args.his_hour)}h_pred{str(self.args.pred_hour)}h", 
            str(self.args.spot_id),
            f"mode{str(self.args.mode)}",
        )
        self.model.load_state_dict(
            torch.load(os.path.join(model_save_path, "checkpoint.pth"))
        )
        criterion = PearsonMSELoss(lambda_pearson=0.4)
        losses = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                loss = criterion(torch.tensor(pred), torch.tensor(true))
                losses.append([loss.item()] + [np.NaN] * (self.args.pred_len - 1))
        # 取平均值
        totla_loss = np.average(losses)
        return totla_loss
    
    def test_save_res(self):
        test_data, test_loader = self._get_data(flag="test")

        print("loading model")
        model_save_path = os.path.join(
            self.args.checkpoints,
            f"his{str(self.args.his_hour)}h_pred{str(self.args.pred_hour)}h", 
            str(self.args.spot_id),
            f"mode{str(self.args.mode)}",
        )
        self.model.load_state_dict(
            torch.load(os.path.join(model_save_path, "checkpoint.pth"))
        )
        criterion = PearsonMSELoss(lambda_pearson=0.4)

        preds = []
        trues = []
        losses = []
        # folder_path = os.path.join(
        #     "data/0411/res/res_test",
        #     str(self.args.spot_id),
        #     str(self.args.seq_len),  # 确保 seq_len 是字符串类型
        #     str(self.args.mode),
        # )
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                # 检查是否为空批次以及是否存在NaN值
                if batch_x.size(0) == 0 or torch.isnan(batch_x).any():
                    print("这是一个空批次或包含NaN值的批次", i)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                loss = criterion(torch.tensor(pred), torch.tensor(true))
                losses.append([loss.item()] + [np.NaN] * (self.args.pred_len - 1))
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)
        losses = np.array(losses)
        print("preds.shape: ", preds.shape)
        print("trues.shape: ", trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("losses.shape: ", losses.shape)
        # losses = losses.reshape(-1, losses.shape[-2], losses.shape[-1])

        # 输出平均PearsonMSELoss
        loss = criterion(torch.tensor(preds), torch.tensor(trues))
        print("PearsonMSELoss:", loss.item())

        # 保存结果
        # 获取时间戳
        times = self.get_time_df_data_test()
        print("times:", len(times))
        print("preds:", len(preds.flatten()))
        print("trues:", len(trues.flatten()))
        print("losses:", len(losses.flatten()))
        # 将时间戳和预测结果保存到CSV文件
        df = pd.DataFrame(
            {
                "time": times,
                "pred": preds.flatten(),
                "real": trues.flatten(),
                "loss": losses.flatten(),
            },
        )
        res_save_path = os.path.join(
            str(self.args.test_res_save_dir),
            str(self.args.spot_id),
            f"mode{str(self.args.mode)}",
        )
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)
        df.to_csv(os.path.join(res_save_path, "result.csv"), index=False)
        return 

    def predict(self, df_raw):
        _, pred_loader = data_provider(self.args, df_raw = df_raw,flag="pred")
        model_load_dir = os.path.join(
            self.args.checkpoints,
            f"his{str(self.args.his_hour)}h_pred{str(self.args.pred_hour)}h", 
            str(self.args.spot_id),
            f"mode{str(self.args.mode)}",
        )
        model_path = model_load_dir + "/" + "checkpoint.pth"
        print("loading model")
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                ).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # 生成未来的时间戳
        pred_times = pd.date_range(
            df_raw.kpi_time.values[-1], periods=self.args.pred_len + 1, freq=self.args.freq
        )
        # 去除第一个时间戳
        pred_times = pred_times[1:]

        # 返回dataframe
        df_res = pd.DataFrame(
            {   
                "spot_id": self.args.spot_id,
                "kpi_time": pred_times,
                "kpi_value": preds.flatten().astype(int),
            }
        )
        
        return df_res


    def set_root_path(self):
        if self.args.data_multi_basedir:
            self.args.root_path = f"{self.args.data_multi_basedir}/{self.args.spot_id}/mode{self.args.mode}"
        
        