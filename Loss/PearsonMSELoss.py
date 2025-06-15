import torch
import numpy as np


class PearsonMSELoss(torch.nn.Module):
    def __init__(self, lambda_pearson=0.2):
        """
        初始化 PearsonMSELoss 类
        :param lambda_pearson: 控制 Pearson 项的权重，默认值为 0.2
        """
        super().__init__()
        self.lambda_pearson = lambda_pearson  # 控制 Pearson 项的权重
        self.mse_loss = torch.nn.MSELoss()  # MSE 损失计算

    def forward(self, y_pred, y_true):
        """
        前向传播方法，计算总损失
        :param y_pred: 模型的预测值
        :param y_true: 真实值
        :return: 总损失
        """
        # --------------------- 计算 MSE 损失 ---------------------
        mse = self.mse_loss(y_pred, y_true)  # 直接调用 PyTorch 的 MSE

        # --------------------- 计算 Pearson 损失项 ---------------------
        # 中心化数据（减去均值）
        y_pred_centered = y_pred - torch.mean(y_pred)
        y_true_centered = y_true - torch.mean(y_true)

        # 计算协方差（分子）
        covariance = torch.sum(y_pred_centered * y_true_centered)

        # 计算分母（两个标准差乘积）
        std_pred = torch.std(y_pred_centered)
        std_true = torch.std(y_true_centered)
        denominator = std_pred * std_true

        # 处理分母为 0 的情况（当预测值或真实值全为常数时）
        if denominator == 0:
            pearson = 0.0  # 无相关性
        else:
            pearson = covariance / denominator

        # 将 Pearson 转换为损失项（1 - rho，范围 [0, 2]）
        loss_pearson = 1 - pearson  # 当 pearson=1（完美正相关）时，损失为 0

        # --------------------- 总损失 ---------------------
        total_loss = mse + self.lambda_pearson * loss_pearson
        return total_loss


def pearson_mse_loss_xgb_train(preds, dtrain):
    """
    计算 PearsonMSE 损失函数的一阶导数（梯度）和二阶导数（海森矩阵）
    :param preds: 模型的预测值
    :param dtrain: XGBoost 的 DMatrix 对象，包含训练数据和标签
    :return: 一阶导数和二阶导数
    """
    # 从 DMatrix 中获取真实标签
    y_true = dtrain.get_label()
    y_pred = np.asarray(preds)
    y_true = np.asarray(y_true)

    # 扁平化 y_pred 和 y_true 以确保形状一致
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # 检查扁平化后的形状是否一致
    if y_pred_flat.shape != y_true_flat.shape:
        raise ValueError(
            f"预测值和真实标签扁平化后的形状不一致: {y_pred_flat.shape} vs {y_true_flat.shape}"
        )

    # 复用类中的 lambda_pearson 值，避免硬编码
    lambda_pearson = 0.2

    # 提前计算均值
    y_pred_mean = np.mean(y_pred_flat)
    y_true_mean = np.mean(y_true_flat)

    # 计算中心化数据
    y_pred_centered = y_pred_flat - y_pred_mean
    y_true_centered = y_true_flat - y_true_mean

    # 提前计算方差和标准差
    var_pred = np.var(y_pred_centered, ddof=0)
    var_true = np.var(y_true_centered, ddof=0)
    std_pred = np.sqrt(var_pred)
    std_true = np.sqrt(var_true)

    # 计算协方差
    covariance = np.sum(y_pred_centered * y_true_centered)

    # 更安全地处理分母为零的情况
    eps = 1e-8
    denominator = std_pred * std_true + eps
    pearson = covariance / denominator

    # 计算 MSE 损失及其导数
    mse_grad = 2 * (y_pred_flat - y_true_flat)
    mse_hess = 2 * np.ones_like(y_pred_flat)

    # 计算 Pearson 损失的梯度
    n = len(y_pred_flat)
    var_pred_eps = var_pred + eps
    denominator_grad = n * std_pred * std_true + eps
    pearson_grad = (y_true_centered / denominator_grad) - (
        pearson * (y_pred_centered / (n * var_pred_eps))
    )
    pearson_hess = np.zeros_like(y_pred_flat)  # 简化处理，实际计算较复杂，这里设为 0

    # 总梯度和总海森矩阵
    total_grad = mse_grad + lambda_pearson * (-pearson_grad)
    total_hess = mse_hess + lambda_pearson * pearson_hess

    # 确保形状符合 XGBoost 2.1.0+ 版本要求
    if len(total_grad.shape) == 1:
        total_grad = total_grad.reshape(-1, 1)
    if len(total_hess.shape) == 1:
        total_hess = total_hess.reshape(-1, 1)

    return total_grad, total_hess


# ... 已有代码 ...


def pearson_mse_loss_xgb_test(y_pred, y_true, lambda_pearson=0.4):
    """
    计算测试集上的 PearsonMSE 损失值
    :param y_pred: 模型在测试集上的预测值，可以是 numpy 数组或 torch.Tensor
    :param y_true: 测试集的真实标签，可以是 numpy 数组或 torch.Tensor
    :param lambda_pearson: 控制 Pearson 项的权重，默认值为 0.4
    :return: 测试集上的 PearsonMSE 损失标量值
    """
    # # 统一转换为 numpy 数组
    # if isinstance(y_pred, torch.Tensor):
    #     y_pred = y_pred.cpu().detach().numpy()
    # if isinstance(y_true, torch.Tensor):
    #     y_true = y_true.cpu().detach().numpy()

    # 扁平化处理
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # 计算 MSE 损失
    mse = np.mean((y_pred_flat - y_true_flat) ** 2)

    # 计算均值
    y_pred_mean = np.mean(y_pred_flat)
    y_true_mean = np.mean(y_true_flat)

    # 计算中心化数据
    y_pred_centered = y_pred_flat - y_pred_mean
    y_true_centered = y_true_flat - y_true_mean

    # 计算协方差
    covariance = np.sum(y_pred_centered * y_true_centered)

    # 计算标准差
    std_pred = np.std(y_pred_flat)
    std_true = np.std(y_true_flat)

    # 处理分母为零的情况
    eps = 1e-8
    denominator = std_pred * std_true + eps
    pearson = covariance / denominator

    # 计算 Pearson 损失
    loss_pearson = 1 - pearson

    # 计算总损失
    total_loss = mse + lambda_pearson * loss_pearson

    return total_loss
