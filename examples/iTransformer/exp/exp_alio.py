from data_provider.data_factory_alio import data_provider
from exp.exp_main import Exp_Main
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_overlap, visual_multi_step
from utils.metrics import metric, CONSISTENCY, CONSISTENCY_MSE, TAM
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
import yaml
import pandas as pd
from awl import AutomaticWeightedLoss
from alio import AliO

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
        data_set, data_loader = data_provider(self.args, flag, lag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.awl is None:
            return super(Exp_AliO, self)._select_optimizer()
        model_optim = optim.Adam([
            {'params': self.model.parameters(), 'lr': self.args.learning_rate},
            {'params': self.awl.parameters(), 'lr': self.args.learning_rate, 'weight_decay': 0}
        ])
        return model_optim
    
    def test_during_training(self, test_data, test_loader, criterion):
        total_loss = []
        self.model.eval()
        if self.awl is not None:
            self.awl.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False, desc="Test"):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs # .detach().cpu()
                true = batch_y # .detach().cpu()

                loss = criterion(pred, true)
                
                loss = loss.detach().cpu()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        if self.awl is not None:
            self.awl.train()
        return total_loss
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        vali_losses = {
            'forecast': [],
            'time-alio': [],
            'freq-alio': [],
        }
        self.model.eval()
        if self.awl is not None:
            self.awl.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), total=len(vali_loader), leave=False, desc="Vali"):
                # batch_x: (batch_size, num_samples, seq_len, num_features)
                batch_size = batch_x.size(0)
                num_samples = batch_x.size(1)
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # Reshape the input and target tensors
                batch_x = batch_x.view(batch_size * num_samples, batch_x.size(2), batch_x.size(3))
                batch_y = batch_y.view(batch_size * num_samples, batch_y.size(2), batch_y.size(3))
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    pass
                else:
                    batch_x_mark = batch_x_mark.view(batch_size * num_samples, batch_x_mark.size(2), batch_x_mark.size(3))
                    batch_y_mark = batch_y_mark.view(batch_size * num_samples, batch_y_mark.size(2), batch_y_mark.size(3))

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs # .detach().cpu()
                true = batch_y # .detach().cpu()

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
                
                loss = loss.detach().cpu()
                
                vali_losses['forecast'].append(forecast_loss.item())
                vali_losses['time-alio'].append(time_alio_loss.item())
                vali_losses['freq-alio'].append(freq_alio_loss.item())
                total_loss.append(loss)
        vali_losses = {k: np.mean(v) for k, v in vali_losses.items()}
        total_loss = np.average(total_loss)
        self.model.train()
        if self.awl is not None:
            self.awl.train()
        return vali_losses, total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train', lag=self.args.alio_lag)
        vali_data, vali_loader = self._get_data(flag='val', lag=self.args.alio_lag)
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}: training', total=len(train_loader), leave=False):
                # batch_x: (batch_size, num_samples, seq_len, num_features)
                batch_size = batch_x.size(0)
                num_samples = batch_x.size(1)
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # Reshape the input and target tensors
                batch_x = batch_x.view(batch_size * num_samples, batch_x.size(2), batch_x.size(3))
                batch_y = batch_y.view(batch_size * num_samples, batch_y.size(2), batch_y.size(3))
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    pass
                else:
                    batch_x_mark = batch_x_mark.view(batch_size * num_samples, batch_x_mark.size(2), batch_x_mark.size(3))
                    batch_y_mark = batch_y_mark.view(batch_size * num_samples, batch_y_mark.size(2), batch_y_mark.size(3))

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
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
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
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
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_losses = {k: np.mean(v) for k, v in train_losses.items()}
            
            vali_losses, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.test_during_training(test_data, test_loader, criterion)
            
            train_forecast_loss = train_losses['forecast']
            train_time_alio_loss = train_losses['time-alio']
            train_freq_alio_loss = train_losses['freq-alio']
            vali_forecast_loss = vali_losses['forecast']
            vali_time_alio_loss = vali_losses['time-alio']
            vali_freq_alio_loss = vali_losses['freq-alio']
            # test_loss = test_loss
            
            print(">> Epoch: {0}, train forecast loss: {1:.7f}, train time-alio loss: {2:.7f}, train freq-alio loss: {3:.7f}".format(epoch + 1, train_forecast_loss, train_time_alio_loss, train_freq_alio_loss))
            print(">> Epoch: {0}, vali forecast loss: {1:.7f}, vali time-alio loss: {2:.7f}, vali freq-alio loss: {3:.7f}".format(epoch + 1, vali_forecast_loss, vali_time_alio_loss, vali_freq_alio_loss))
            print(">> Epoch: {0}, vali total loss: {1:.7f}".format(epoch + 1, vali_total_loss))
            # print(">> Epoch: {0}, test loss: {1:.7f}".format(epoch + 1, test_loss))
            
            early_stopping(vali_total_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
