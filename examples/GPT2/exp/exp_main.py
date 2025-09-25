from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import GPT4TS
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_overlap, visual_multi_step
from utils.metrics import metric, CONSISTENCY, CONSISTENCY_MSE, TAM

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
import yaml
import pandas as pd
import sys

SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.model = GPT4TS.Model(args, self.device)

    def _build_model(self):
        model_dict = {
            'GPT4TS': GPT4TS,
        }
        model = model_dict[self.args.model].Model(self.args, self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _select_scheduler(self, model_optim):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        return scheduler


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.in_layer.eval()
        self.model.out_layer.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                        pred = outputs.detach().cpu()
                        true = batch_y.detach().cpu()

                        loss = criterion(pred, true)
                        total_loss.append(loss)
                else:
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)
                    total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.in_layer.train()
        self.model.out_layer.train()

        return total_loss
            

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.freq != 'h':
            setattr(train_data, 'freq', SEASONALITY_MAP[self.args.freq])
            
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim)
        
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=train_steps, desc=f'Epoch: {epoch} training...', leave=False):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                               
                # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape) 
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)                
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:                    
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = 0 # self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            if self.args.cos:
                scheduler.step()
                print("Learning rate: {}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
            
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        inverse_transform = test_data.inverse_transform
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pdf_results = {}
        pdf_interval = max(len(test_loader) // 20, 1)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

            
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x[:, -self.args.seq_len:, :])
                else:
                    outputs = self.model(batch_x[:, -self.args.seq_len:, :])
            
                # encoder - decoder
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                
                if i % pdf_interval == 0:
                    pdf_results[i] = (true, pred)
                    input = batch_x.detach().cpu().numpy()
                    _gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    _pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    os.makedirs(os.path.join(folder_path, 'prediction'), exist_ok=True)
                    visual(_gt, _pd, os.path.join(folder_path, 'prediction/' + str(i) + '.pdf'))
                               
        if not os.path.exists(f'{folder_path}/overlap'):
            os.makedirs(f'{folder_path}/overlap')
        
        # Multiprocessing으로 figure 생성. CPU OOM 주의
        tasks = []
        for i in pdf_results.keys():
            gt, pred = pdf_results[i]
            channel_interval = max(gt.shape[2] // 10, 1)
            for channel in range(0, gt.shape[2], channel_interval):
                one_channel_gt = gt[:, :, channel]
                one_channel_pred = pred[:, :, channel]
                tasks.append((
                    one_channel_gt, one_channel_pred, channel, folder_path, i, self.args.alio_test_num_samples
                ))
        
        num_workers = 4
        with Pool(processes=num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(generate_figure, tasks), total=len(tasks), desc='Generating Figures...'):
                pass
        print(">> Figure generating is done")
        print(">> Figure generating is done", file=sys.stderr)
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(">> Computing metrics...")
        print(">> Computing metrics...", file=sys.stderr)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        consistency = CONSISTENCY(preds)
        consistency_mse = CONSISTENCY_MSE(preds)
        
        inverse_consistency = TAM(consistency)
        inverse_consistency_mse = TAM(consistency_mse)
        print('>>>> mse:{}\n>>>> mae:{}\n>>>> consistency:{}\n>>>> consistency_mse:{}\n>>>> inverse_consistency:{}\n>>>> inverse_consistency_mse:{}'.format(mse, mae, consistency, consistency_mse, inverse_consistency, inverse_consistency_mse))
        print('>>>> mse:{}\n>>>> mae:{}\n>>>> consistency:{}\n>>>> consistency_mse:{}\n>>>> inverse_consistency:{}\n>>>> inverse_consistency_mse:{}'.format(mse, mae, consistency, consistency_mse, inverse_consistency, inverse_consistency_mse), file=sys.stderr)
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, consistency:{}, consistency_mse:{}, inverse_consistency:{}, inverse_consistency_mse:{}  \n'.format(mse, mae, consistency, consistency_mse, inverse_consistency, inverse_consistency_mse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, consistency, consistency_mse, inverse_consistency, inverse_consistency_mse]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        
        with open(folder_path + 'config.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
            
        df = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'consistency': consistency,
            'consistency_mse': consistency_mse,
            'inverse_consistency': inverse_consistency,
            'inverse_consistency_mse': inverse_consistency_mse
        }
        
        df = pd.DataFrame(df, index=[0])
        df.to_csv(folder_path + 'metrics.csv', index=False)

        return
                
                

def generate_figure(data):
    gt, pred, channel, folder_path, index, num_samples = data
    visual_overlap(gt, pred, os.path.join(folder_path, 'overlap', 'channel_' + str(channel) + '_batch_' + str(index) + '_multi.pdf'), num_samples=num_samples)
