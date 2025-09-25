import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_alio import Exp_AliO
from exp.exp_alio_finetune import Exp_AliOFT
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
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
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # CycleNet.
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # SegRNN
    parser.add_argument('--rnn_type', default='gru', help='rnn_type')
    parser.add_argument('--dec_way', default='pmf', help='decode way')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length')
    parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

    # SparseTSF
    parser.add_argument('--period_len', type=int, default=24, help='period_len')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
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
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    parser.add_argument('--alio_time_weight', type=float, default=0.0, help='time domain weight in alio')
    parser.add_argument('--alio_time_loss', type=str, default='mse', help='time domain loss in alio')
    parser.add_argument('--alio_freq_weight', type=float, default=0.0, help='frequency domain weight in alio')
    parser.add_argument('--alio_freq_loss', type=str, default='mse', help='frequency domain loss in alio')
    parser.add_argument('--alio_lag', type=int, default=0, help='lag in alio')
    parser.add_argument('--alio_num_samples', type=int, default=1, help='number of samples in alio')
    parser.add_argument('--alio_awl', type=str, default='off', help='whether to use awl in alio')
    parser.add_argument('--alio_debug', action='store_true', help='If you true on this option, you can use alio_time_weight = 0 and alio_freq_weight = 0')
    parser.add_argument('--alio_test_num_samples', type=int, default=5, help='number of samples in alio test')
    parser.add_argument('--alio_awl_init', type=str, default='sqrt', help='initial value of awl')

    parser.add_argument('--freq_reg', action='store_true')
    parser.add_argument('--fast_dataloader', action='store_true', help='Load all dataset to specific `device`')
    parser.add_argument('--data_num', type=int, default=559, help='number of data file')

    parser.add_argument('--finetune', action='store_true', help='whether to use fine-tune')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    exp_mode = 'basic' if args.alio_num_samples == 1 else 'alio'
    
    assert args.alio_time_weight >= 0, f"alio_time_weight should be greater than or equal to 0, but got {args.alio_time_weight}"
    assert args.alio_freq_weight >= 0, f"alio_freq_weight should be greater than or equal to 0, but got {args.alio_freq_weight}"
    assert args.alio_lag >= 0, f"alio_lag should be greater than or equal to 0, but got {args.alio_lag}"
    assert args.alio_num_samples >= 1, f"alio_num_samples should be greater than or equal to 1, but got {args.alio_num_samples}"
    assert args.itr == 1, f"alio only supports one iteration, but got {args.itr}. You should use different random seeds for different iterations."

    if exp_mode == 'basic':
        assert args.alio_time_weight == 0 and args.alio_freq_weight == 0, "In basic mode, time_weight and freq_weight should be 0"
        assert args.alio_lag == 0, "In basic mode, lag should be 0"
        assert args.alio_awl == 'off', "In basic mode, awl should be off"
    else:
        if not args.alio_debug:
            assert args.alio_time_weight > 0 or args.alio_freq_weight > 0, "At least one of time_weight and freq_weight should be greater than 0\nIf you want to use alio without time and freq domain loss, please use alio_debug option"
        assert args.alio_lag > 0, "lag should be greater than 0"

    if '_' in args.model_id:
        setattr(args, 'model_id', args.model_id.replace('_', '-'))

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main if exp_mode == 'basic' else Exp_AliO
    if args.finetune and exp_mode != 'basic':
        Exp = Exp_AliOFT

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            
            if args.alio_awl == 'on':
                exp_mode = f'{exp_mode}-awl'
            if args.freq_reg:
                exp_mode = f'{exp_mode}-freq'
            if args.finetune:
                exp_mode = f'{exp_mode}-finetune'
            exp_mode = f'{args.features}-{exp_mode}'

            # setting record of experiments
            
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

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        raise NotImplementedError('>> Test mode is not implemented yet')
        # ii = 0
        # setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}_seed{}'.format(
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.pred_len,
        #     args.cycle,
        #     args.model_type,
        #     fix_seed)

        # exp = Exp(args)  # set experiments
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        # torch.cuda.empty_cache()
