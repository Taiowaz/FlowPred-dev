import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.config import get_args
from skopt import gp_minimize
from skopt.space import Categorical

def objective(params):
    dynamic_dim, hidden_dim, hidden_layers, num_blocks, alpha = params
    args = get_args()

    args.root_path = "data/pred_6h_args/ogn/24/288_72/mode_1"
    args.mode = 1
    # 设置超参数
    args.dynamic_dim = dynamic_dim
    args.hidden_dim = hidden_dim
    args.hidden_layers = hidden_layers
    args.num_blocks = num_blocks
    args.alpha = alpha

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(" ", "")
            device_ids = args.devices.split(",")
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    Exp = Exp_Main
    valid_metrics = []

    if args.is_training:
        for ii in range(args.itr):
            setting = "{}_{}_{}_ft{}_sl{}_pl{}_segl{}_dyna{}_h{}_l{}_nb{}_a{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.seg_len,
                args.dynamic_dim,
                args.hidden_dim,
                args.hidden_layers,
                args.num_blocks,
                args.alpha,
                args.des,
                ii,
            )

            exp = Exp(args)
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)
            # 假设 exp.test 返回一个评估指标，如损失值
            metric = exp.test(setting)
            print(f"Metric for setting {setting}: {metric}")  # 添加日志输出
            # 检查是否为有效指标
            if not np.isnan(metric) and not np.isinf(metric):
                valid_metrics.append(metric)
            else:
                print(f"Warning: Invalid metric encountered for setting {setting}. Skipping this result.")
            torch.cuda.empty_cache()
    
    if valid_metrics:
        return np.mean(valid_metrics)
    else:
        return np.finfo(float).max

def main():
    # 定义超参数搜索空间，使用 Categorical 类设置可选参数
    space = [
        Categorical([64, 128, 256, 512], name='dynamic_dim'),
        Categorical([32, 64, 128, 256], name='hidden_dim'),
        Categorical([1, 2, 3, 4], name='hidden_layers'),
        Categorical([2, 3, 4, 5], name='num_blocks'),
        Categorical([0.1, 0.2, 0.3, 0.4], name='alpha')
    ]

    result = gp_minimize(objective, space, n_calls=10)
    best_params = {
        "dynamic_dim": result.x[0],
        "hidden_dim": result.x[1],
        "hidden_layers": result.x[2],
        "num_blocks": result.x[3],
        "alpha": result.x[4]
    }
    best_metric = result.fun

    print("Best hyperparameters:", best_params)
    print("Best metric:", best_metric)

if __name__ == "__main__":
    main()
