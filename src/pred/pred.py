from src.utils.utils_data import get_spot_config
from src.utils.config import parse_and_initialize_args
from src.exp.exp_main import Exp_Main
from src.pattern.pattern_pred import pattern_prediction
import torch
import datetime

def predict(spot_id,his_hour,pred_hour,input_df):
    # 记录开始时间
    start_time = datetime.datetime.now()
    print(f">>>>>>>开始预测时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}<<<<<<<<<")
    
    df_raw = input_df
    pred_len = int(pred_len)
    his_len = int(his_len)
    args = parse_and_initialize_args()
    args.is_training = 0
    args.spot_id = spot_id
    args.freq, args.his_len, args.pred_len = get_spot_config(spot_id, his_hour, pred_hour)
    args.mode=pattern_prediction(df_raw)
    args.checkpoints = f"aux_data/checkpoints/{his_hour}_{pred_hour}"

    print("Args in Predict:")
    print(args)
    exp = Exp_Main(args)  # set experiments
    setting = "his{}_pred{}_freq{}".format(
        his_hour,
        pred_hour,
        args.freq,
    )
    print(">>>>>>>spot_{} predict : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(args.spot_id,setting))
    df_pred = exp.predict(setting=setting,df_raw=df_raw)
    torch.cuda.empty_cache()
    
    # 记录结束时间
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f">>>>>>>预测结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}<<<<<<<<<")
    print(f">>>>>>>预测总耗时: {duration}<<<<<<<<<")
    return df_pred