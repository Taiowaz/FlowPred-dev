from ast import arg
from .data_loader import Dataset_flow, Dataset_Pred
from torch.utils.data import DataLoader


def data_provider(args, flag, df_raw=None):
    Data = Dataset_flow
    timeenc = 0 if args.embed != "timeF" else 1

    if flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    # 创建数据集
    if flag == "pred":
        data_set = Data(
            df_raw=df_raw,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            datadir_flag=False if args.datadir_flag == "False" else True,
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
