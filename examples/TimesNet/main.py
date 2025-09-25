import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_alio import Exp_AliO
import random
import numpy as np
import time
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")


    # AliO
    parser.add_argument('--random_seed', type=int, default=None, help='random seed', required=True, choices=[2025, 2024, 2023, 2022, 2021])
    parser.add_argument('--alio_time_weight', type=float, default=0.0, help='time domain weight in AliO')
    parser.add_argument('--alio_time_loss', type=str, default='mse', help='time domain loss in AliO')
    parser.add_argument('--alio_freq_weight', type=float, default=0.0, help='frequency domain weight in AliO')
    parser.add_argument('--alio_freq_loss', type=str, default='mse', help='frequency domain loss in AliO')
    parser.add_argument('--alio_lag', type=int, default=0, help='lag in AliO')
    parser.add_argument('--alio_num_samples', type=int, default=1, help='number of samples in AliO')
    parser.add_argument('--alio_awl', type=str, default='off', help='whether to use awl in AliO')
    parser.add_argument('--alio_debug', action='store_true', help='If you true on this option, you can use alio_time_weight = 0 and alio_freq_weight = 0')
    parser.add_argument('--alio_test_num_samples', type=int, default=5, help='number of samples in AliO test')
    parser.add_argument('--alio_awl_init', type=str, default='sqrt', help='initial value of awl')

    parser.add_argument('--freq_reg', action='store_true')
    parser.add_argument('--fast_dataloader', action='store_true', help='Load all dataset to specific `device`')

    args = parser.parse_args()

    # AliO
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    exp_mode = 'basic' if args.alio_num_samples == 1 else 'alio'
    
    assert args.alio_time_weight >= 0, f"alio_time_weight should be greater than or equal to 0, but got {args.alio_time_weight}"
    assert args.alio_freq_weight >= 0, f"alio_freq_weight should be greater than or equal to 0, but got {args.alio_freq_weight}"
    assert args.alio_lag >= 0, f"alio_lag should be greater than or equal to 0, but got {args.alio_lag}"
    assert args.alio_num_samples >= 1, f"alio_num_samples should be greater than or equal to 1, but got {args.alio_num_samples}"
    assert args.itr == 1, f"AliO only supports one iteration, but got {args.itr}. You should use different random seeds for different iterations."

    if exp_mode == 'basic':
        assert args.alio_time_weight == 0 and args.alio_freq_weight == 0, "In basic mode, time_weight and freq_weight should be 0"
        assert args.alio_lag == 0, "In basic mode, lag should be 0"
        assert args.alio_awl == 'off', "In basic mode, awl should be off"
    else:
        if not args.alio_debug:
            assert args.alio_time_weight > 0 or args.alio_freq_weight > 0, "At least one of time_weight and freq_weight should be greater than 0\nIf you want to use AliO without time and freq domain loss, please use alio_debug option"
        assert args.alio_lag > 0, "lag should be greater than 0"

    if '_' in args.model_id:
        setattr(args, 'model_id', args.model_id.replace('_', '-'))
    
    
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main if exp_mode == 'basic' else Exp_AliO
    
    print('\n\n')
    if exp_mode == 'basic':
        print('>> Basic experiment')
        print(f'>>>> random_seed={args.random_seed}')
    else:
        print('>> AliO experiment')
        print(f'>>>> random_seed={args.random_seed}')
        print(f'>>>> num_samples={args.alio_num_samples}')
        print(f'>>>> alio_lag={args.alio_lag}')
        print(f'>>>> alio_awl={args.alio_awl}')
        print(f'>>>> alio_time_weight={args.alio_time_weight}')
        print(f'>>>> alio_freq_weight={args.alio_freq_weight}')
        print(f'>>>> alio_time_loss={args.alio_time_loss}')
        print(f'>>>> alio_freq_loss={args.alio_freq_loss}')
    print('\n\n')
    
    if args.is_training:
        # setting record of experiments
        
        if args.alio_awl == 'on':
            exp_mode = f'{exp_mode}-awl'
        if args.freq_reg:
            exp_mode = f'{exp_mode}-freq'
        exp_mode = f'{args.features}-{exp_mode}'
        setting = f"{exp_mode}/" \
        f"{args.model_id}_" \
        f"{args.pred_len}_" \
        f"{args.model}_" \
        f"{args.data}_" \
        f"{args.alio_num_samples}_" \
        f"{args.alio_lag}_" \
        f"{args.alio_awl}_" \
        f"{args.alio_time_weight}_" \
        f"{args.alio_freq_weight}_" \
        f"{args.alio_time_loss}_" \
        f"{args.alio_freq_loss}_" \
        f"{args.random_seed}"
        
        print(f'>> Check setting: {setting}')
        if os.path.exists(f'./results/{exp_mode}/') is False:
            os.makedirs(f'./results/{exp_mode}/')
        if os.path.exists(f'./results/{setting}/metrics.csv') is True or os.path.exists(f'./results/{setting}/metrics.npy') is True:
            raise FileExistsError(f'>> Setting {setting} already exists in ./results/{setting}\n>> To avoid overwrite, please change model_id or random_seed')

        print(f'>> Start training with setting: {setting}')
        exp = Exp(args)  # set experiments
        exp.train(setting)

        
        print(f'>> Start testing with setting: {setting}')
        delay_time = time.time()
        delay_time = int(delay_time % 150 + 30)
        print(f'>> Delay time: {delay_time} seconds for preventing bottleneck')
        time.sleep(delay_time)
        exp.test(setting)

        torch.cuda.empty_cache()
    else:
        if args.alio_awl == 'on':
            exp_mode = f'{exp_mode}-awl'
        if args.freq_reg:
            exp_mode = f'{exp_mode}-freq'
        exp_mode = f'{args.features}-{exp_mode}'
        setting = f"{exp_mode}/" \
        f"{args.model_id}_" \
        f"{args.pred_len}_" \
        f"{args.model}_" \
        f"{args.data}_" \
        f"{args.alio_num_samples}_" \
        f"{args.alio_lag}_" \
        f"{args.alio_awl}_" \
        f"{args.alio_time_weight}_" \
        f"{args.alio_freq_weight}_" \
        f"{args.alio_time_loss}_" \
        f"{args.alio_freq_loss}_" \
        f"{args.random_seed}"

        exp = Exp(args)  # set experiments
        print(f'>> Start testing with setting: {setting}')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()