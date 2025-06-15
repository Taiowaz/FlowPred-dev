from run import parse_and_initialize_args
from exp.exp_main import Exp_Main
from encode.encode_pred import encode_prediction
import torch

def predict(df_raw):
    args = parse_and_initialize_args()
    args.is_training = 0
    args.spot_id = df_raw['spot_id'].iloc[0]
    args.mode=encode_prediction(df_raw)
    args.seq_len = 288
    args.pred_len = 24
    args.freq = "5min"
    args.root_path = f"data/ogn/mode/{args.spot_id}/{args.seq_len}_{args.pred_len}/mode_{args.mode}"
    Exp = Exp_Main

    setting = "{}_{}_{}_dyna{}_h{}_l{}_nb{}_a{}".format(
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.dynamic_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.num_blocks,
        args.alpha,
    )
    exp = Exp(args)  # set experiments
    print(">>>>>>>spot_{} predict : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(args.spot_id,setting))
    df_pred = exp.predict(setting=setting,df_raw=df_raw)
    torch.cuda.empty_cache()
    # 输出测试完毕的信息
    print("predict completed.")
    return df_pred