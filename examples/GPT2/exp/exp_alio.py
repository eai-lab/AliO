from data_provider.data_factory_alio import data_provider
from exp.exp_main import Exp_Main
from models import GPT4TS
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_overlap, visual_multi_step
from utils.metrics import metric, CONSISTENCY, CONSISTENCY_MSE, TAM
from awl import AutomaticWeightedLoss
from alio import AliO

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

class Exp_AliO(Exp_Main):
    def __init__(self, args):
        super(Exp_AliO, self).__init__(args)
        print("Experiment: AliO")
        self.args = args
        if self.args.alio_awl == 'on':
            self.awl = AutomaticWeightedLoss(num=2, verbose=True, args=self.args)
            self.awl.to(self.device)
        elif self.args.alio_awl == 'off':
            self.awl = None
        else:
            raise ValueError("Invalid value for alio_awl")
        self.alio = AliO(num_samples=self.args.alio_num_samples, lag=self.args.alio_lag, args=self.args)

    def _get_data(self, flag, lag=1):
        data_set, data_loader = data_provider(self.args, flag, lag=lag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.awl is None:
            return super(Exp_AliO, self)._select_optimizer()
        model_optim = optim.Adam([
            {'params': self.model.parameters(), 'lr': self.args.learning_rate},
            {'params': self.awl.parameters(), 'lr': self.args.learning_rate, 'weight_decay': 0}
        ])
        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        vali_losses = {
            'forecast': [],
            'time-alio': [],
            'freq-alio': [],
        }
        self.model.in_layer.eval()
        self.model.out_layer.eval()
        if self.awl is not None:
            self.awl.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), total=len(vali_loader), desc='Validation...', leave=False):
                # batch_x: (batch_size, num_samples, seq_len, num_features)
                batch_size = batch_x.size(0)
                num_samples = batch_x.size(1)
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # Reshape the input and target tensors
                batch_x = batch_x.view(batch_size * num_samples, batch_x.size(2), batch_x.size(3))
                batch_y = batch_y.view(batch_size * num_samples, batch_y.size(2), batch_y.size(3))
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                        pred = outputs
                        true = batch_y

                        forecast_loss = criterion(pred, true)
                        pred = pred.view(batch_size, num_samples, outputs.shape[-2], outputs.shape[-1])
                        true = true.view(batch_size, num_samples, batch_y.shape[-2], batch_y.shape[-1])
                        
                        time_alio_loss, freq_alio_loss = self.alio(pred, true)
                        time_alio_loss = time_alio_loss * self.args.alio_time_weight
                        freq_alio_loss = freq_alio_loss * self.args.alio_freq_weight
                        
                        if self.awl is not None:
                            loss = forecast_loss + self.awl(time_alio_loss, freq_alio_loss)
                        else:
                            loss = forecast_loss + time_alio_loss + freq_alio_loss
                else:
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    pred = outputs
                    true = batch_y

                    forecast_loss = criterion(pred, true)
                    pred = pred.view(batch_size, num_samples, outputs.shape[-2], outputs.shape[-1])
                    true = true.view(batch_size, num_samples, batch_y.shape[-2], batch_y.shape[-1])
                    
                    time_alio_loss, freq_alio_loss = self.alio(pred, true)
                    time_alio_loss = time_alio_loss * self.args.alio_time_weight
                    freq_alio_loss = freq_alio_loss * self.args.alio_freq_weight
                    
                    if self.awl is not None:
                        loss = forecast_loss + self.awl(time_alio_loss, freq_alio_loss)
                    else:
                        loss = forecast_loss + time_alio_loss + freq_alio_loss
                    
                forecast_loss = forecast_loss.detach().cpu()
                time_alio_loss = time_alio_loss.detach().cpu()
                freq_alio_loss = freq_alio_loss.detach().cpu()
                vali_losses['forecast'].append(forecast_loss.item())
                vali_losses['time-alio'].append(time_alio_loss.item())
                vali_losses['freq-alio'].append(freq_alio_loss.item())
                total_loss.append(loss.detach().cpu())

        vali_losses = {k: np.mean(v) for k, v in vali_losses.items()}
        total_loss = np.average(total_loss)
        self.model.in_layer.train()
        self.model.out_layer.train()
        if self.awl is not None:
            self.awl.train()

        return vali_losses, total_loss

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
            train_losses = {
                'forecast': [],
                'time-alio': [],
                'freq-alio': [],
            }

            self.model.train()
            if self.awl is not None:
                self.awl.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=train_steps, desc=f'Epoch: {epoch} training...', leave=False):
                # batch_x: (batch_size, num_samples, seq_len, num_features)
                batch_size = batch_x.size(0)
                num_samples = batch_x.size(1)
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                batch_x = batch_x.view(batch_size * num_samples, batch_x.size(2), batch_x.size(3))
                batch_y = batch_y.view(batch_size * num_samples, batch_y.size(2), batch_y.size(3))
                               
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        
                        forecast_loss = criterion(outputs, batch_y)
                        outputs = outputs.view(batch_size, num_samples, outputs.shape[-2], outputs.shape[-1])
                        batch_y = batch_y.view(batch_size, num_samples, batch_y.shape[-2], batch_y.shape[-1])
                        
                        time_alio_loss, freq_alio_loss = self.alio(outputs, batch_y)
                        time_alio_loss = time_alio_loss * self.args.alio_time_weight
                        freq_alio_loss = freq_alio_loss * self.args.alio_freq_weight
                                                
                        if self.awl is not None:
                            awl_loss = self.awl(time_alio_loss, freq_alio_loss)
                            loss = forecast_loss + self.awl(time_alio_loss, freq_alio_loss)
                        else:
                            loss = forecast_loss + time_alio_loss + freq_alio_loss
                else:
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    
                    forecast_loss = criterion(outputs, batch_y)
                    outputs = outputs.view(batch_size, num_samples, outputs.shape[-2], outputs.shape[-1])
                    batch_y = batch_y.view(batch_size, num_samples, batch_y.shape[-2], batch_y.shape[-1])
                    
                    time_alio_loss, freq_alio_loss = self.alio(outputs, batch_y)
                    time_alio_loss = time_alio_loss * self.args.alio_time_weight
                    freq_alio_loss = freq_alio_loss * self.args.alio_freq_weight
                                            
                    if self.awl is not None:
                        loss = forecast_loss + self.awl(time_alio_loss, freq_alio_loss)
                    else:
                        loss = forecast_loss + time_alio_loss + freq_alio_loss

                train_losses['forecast'].append(forecast_loss.item())
                train_losses['time-alio'].append(time_alio_loss.item())
                train_losses['freq-alio'].append(freq_alio_loss.item())

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
            train_losses = {k: np.mean(v) for k, v in train_losses.items()}
            vali_losses, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            
            
            train_forecast_loss = train_losses['forecast']
            train_time_alio_loss = train_losses['time-alio']
            train_freq_alio_loss = train_losses['freq-alio']
            vali_forecast_loss = vali_losses['forecast']
            vali_time_alio_loss = vali_losses['time-alio']
            vali_freq_alio_loss = vali_losses['freq-alio']
            
            print(">> Epoch: {0}, train forecast loss: {1:.7f}, train time-alio loss: {2:.7f}, train freq-alio loss: {3:.7f}".format(epoch + 1, train_forecast_loss, train_time_alio_loss, train_freq_alio_loss))
            print(">> Epoch: {0}, vali forecast loss: {1:.7f}, vali time-alio loss: {2:.7f}, vali freq-alio loss: {3:.7f}".format(epoch + 1, vali_forecast_loss, vali_time_alio_loss, vali_freq_alio_loss))
            print(">> Epoch: {0}, vali total loss: {1:.7f}".format(epoch + 1, vali_total_loss))
            
            if self.args.cos:
                scheduler.step()
                print("Learning rate: {}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            early_stopping(vali_total_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model