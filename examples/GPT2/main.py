import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_alio import Exp_AliO
import random
import numpy as np


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT4TS')

    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
    parser.add_argument('--data_path', type=str, default='traffic.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)

    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)

    parser.add_argument('--itr', type=int, default=3)
    parser.add_argument('--cos', type=int, default=0)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

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

    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)

    args = parser.parse_args()

    # AliO
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    exp_mode = 'basic' if args.alio_num_samples == 1 else 'alio'
    
    if args.freq == 0:
        setattr(args, 'freq', 'h')
    
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

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
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
    if os.path.exists(f'./results/{setting}') is True:
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    # SEASONALITY_MAP = {
    #     "minutely": 1440,
    #     "10_minutes": 144,
    #     "half_hourly": 48,
    #     "hourly": 24,
    #     "daily": 7,
    #     "weekly": 1,
    #     "monthly": 12,
    #     "quarterly": 4,
    #     "yearly": 1
    # }

    # mses = []
    # maes = []

    # for ii in range(args.itr):

    #     setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
    #                                                                     args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
    #                                                                     args.d_ff, args.embed, ii)
    #     path = os.path.join(args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     train_data, train_loader = data_provider(args, 'train')
    #     vali_data, vali_loader = data_provider(args, 'val')
    #     test_data, test_loader = data_provider(args, 'test')

    #     if args.freq != 'h':
    #         args.freq = SEASONALITY_MAP[test_data.freq]
    #         print("freq = {}".format(args.freq))

    #     device = torch.device('cuda:0')

    #     time_now = time.time()
    #     train_steps = len(train_loader)

    #     if args.model == 'PatchTST':
    #         model = PatchTST(args, device)
    #         model.to(device)
    #     elif args.model == 'DLinear':
    #         model = DLinear(args, device)
    #         model.to(device)
    #     else:
    #         model = GPT4TS(args, device)
    #     # mse, mae = test(model, test_data, test_loader, args, device, ii)

    #     params = model.parameters()
    #     model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        
    #     early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    #     if args.loss_func == 'mse':
    #         criterion = nn.MSELoss()
    #     elif args.loss_func == 'smape':
    #         class SMAPE(nn.Module):
    #             def __init__(self):
    #                 super(SMAPE, self).__init__()
    #             def forward(self, pred, true):
    #                 return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    #         criterion = SMAPE()
        
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    #     for epoch in range(args.train_epochs):

    #         iter_count = 0
    #         train_loss = []
    #         epoch_time = time.time()
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

    #             iter_count += 1
    #             model_optim.zero_grad()
    #             batch_x = batch_x.float().to(device)

    #             batch_y = batch_y.float().to(device)
    #             batch_x_mark = batch_x_mark.float().to(device)
    #             batch_y_mark = batch_y_mark.float().to(device)
                
    #             outputs = model(batch_x, ii)

    #             outputs = outputs[:, -args.pred_len:, :]
    #             batch_y = batch_y[:, -args.pred_len:, :].to(device)
    #             loss = criterion(outputs, batch_y)
    #             train_loss.append(loss.item())

    #             if (i + 1) % 1000 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()
    #             loss.backward()
    #             model_optim.step()

            
    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

    #         train_loss = np.average(train_loss)
    #         vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
    #         # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
    #         # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
    #         #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
    #             epoch + 1, train_steps, train_loss, vali_loss))

    #         if args.cos:
    #             scheduler.step()
    #             print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    #         else:
    #             adjust_learning_rate(model_optim, epoch + 1, args)
    #         early_stopping(vali_loss, model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     model.load_state_dict(torch.load(best_model_path))
    #     print("------------------------------------")
    #     mse, mae = test(model, test_data, test_loader, args, device, ii)
    #     mses.append(mse)
    #     maes.append(mae)

    # mses = np.array(mses)
    # maes = np.array(maes)
    # print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    # print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))