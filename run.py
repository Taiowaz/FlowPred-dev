import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.config import get_args

def main():
    args = get_args()

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

    print("Args in experiment:")
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
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

            exp = Exp(args)  # set experiments
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)

            # print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            # exp.test(setting)

            # if args.do_predict:
            #     print(
            #         ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
            #             setting
            #         )
            #     )
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
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

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test_save_res(setting, test=1)
        torch.cuda.empty_cache()
        # 输出测试完毕的信息
        print("Testing completed.")

if __name__ == "__main__":
    main()
