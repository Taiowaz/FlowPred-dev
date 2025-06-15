import argparse

def get_args():
    """
    设置命令行参数并返回解析后的参数对象。

    Returns:
        argparse.Namespace: 解析后的命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="Koopa for Time Series Forecasting")

    # basic config
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="Flow_288_24", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="Koopa",
        help="model name, options: [Koopa]",
    )

    # data loader
    parser.add_argument("--data", type=str, default="Flow", help="dataset type")
    parser.add_argument(
        "--datadir_flag",
        type=str,
        default=True,
        help="A boolean flag to indicate whether to read data from a directory. Default is True.",
    )
    parser.add_argument(
        "--root_path", type=str, default="data/pred_6h_args/ogn/24/288_72/mode_0", help="root path of the data file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="14100_preprocessed.csv",
        help="data file,if datadir_flag is False , data_path is the data file name, else data_path is the directory name",
    )

    # 添加 mask_spectrum 保存和加载路径参数
    parser.add_argument(
        "--mask_spectrum_dir",
        type=str,
        default="maskspectrum",
        help="Path to save/load the mask spectrum during training"
    )

    parser.add_argument(
        "--test_res_save_dir",
        type=str,
        default="data/pred_6h_args/res/res_test",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="S",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="kpi_value", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="5min",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="data/pred_6h_args/res/model",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument(
        "--spot_id",
        type=str,
        default="14100",
    )
    parser.add_argument("--mode", type=str, default="0")
    parser.add_argument("--seq_len", type=int, default=288, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=24, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=72, help="prediction sequence length"
    )

    # model define
    parser.add_argument("--enc_in", type=int, default=1, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=1, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=1, help="output size")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="whether to predict unseen future data"
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="PearsonMSELoss", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0", help="device ids of multile gpus"
    )
    parser.add_argument("--seed", type=int, default=2023, help="random seed")

    # Koopa
    parser.add_argument(
        "--dynamic_dim", type=int, default=128, help="latent dimension of koopman embedding"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="hidden dimension of en/decoder"
    )
    parser.add_argument(
        "--hidden_layers", type=int, default=2, help="number of hidden layers of en/decoder"
    )
    parser.add_argument(
        "--seg_len", type=int, default=48, help="segment length of time series"
    )
    parser.add_argument("--num_blocks", type=int, default=3, help="number of Koopa blocks")
    parser.add_argument("--alpha", type=float, default=0.2, help="spectrum filter ratio")
    parser.add_argument(
        "--multistep",
        action="store_true",
        help="whether to use approximation for multistep K",
        default=False,
    )

    args = parser.parse_args()
    return args
