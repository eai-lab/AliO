import torch
import torch.nn as nn
import torch.nn.functional as F

from awl import AutomaticWeightedLoss

class AliO(nn.Module):
    def __init__(self, num_samples, lag=1, args=None, **kwargs):
        """
        Align Common Outputs
        
        alio_time_weight and alio_freq_weight is not calculated in this class. This is only for weather compute the time (freq) domain loss or not.

        Args:
            num_samples (_type_): _description_
            lag (int, optional): _description_. Defaults to 1.
            args (_type_, optional): _description_. Defaults to None.
        """
        super(AliO, self).__init__()

        assert num_samples > 1, f"num_samples should be greater than 1, but got {num_samples}"
        assert lag > 0, f"lag should be greater than 0, but got {lag}"
        assert args is not None, "args should be provided"
        
        self.num_samples = num_samples
        self.lag = lag
        self.args = args
        self.kwargs = kwargs

        self.time_weight = args.alio_time_weight
        self.freq_weight = args.alio_freq_weight
        assert self.time_weight > 0 or self.freq_weight > 0, "At least one of time_weight and freq_weight should be greater than 0"
        
        self.time_loss = nn.MSELoss() if args.alio_time_loss == "mse" else nn.L1Loss()
        print(f"AliO: time_weight={self.time_weight}, freq_weight={self.freq_weight}, time_loss={args.alio_time_loss}, freq_loss={args.alio_freq_loss}")

        if self.time_weight == 0:
            print("AliO: time domain loss is not calculated")
        if self.freq_weight == 0:
            print("AliO: freq domain loss is not calculated")
        self.step = 0
        
        
    def freq_forward(self, pred1, pred2, targets):
        pred1 = torch.fft.fft(pred1, dim=-2, norm='forward')
        pred2 = torch.fft.fft(pred2, dim=-2, norm='forward')
        targets = torch.fft.fft(targets, dim=-2, norm='forward')
        
        with torch.no_grad():
            compare1 = torch.abs(pred1 - targets)
            compare2 = torch.abs(pred2 - targets)
            mask = compare1 > compare2
        
        # Since the nn.MSELoss or nn.L1Loss do not support complex number, we need to manually calculate the loss
        power = 2 if self.args.alio_freq_loss == 'mse' else 1
            
        loss1 = torch.sum(torch.abs(pred1[mask] - pred2[mask].detach()).pow(power))
        loss2 = torch.sum(torch.abs(pred2[~mask] - pred1[~mask].detach()).pow(power))
        
        if torch.isnan(loss1).any():
            loss1 = torch.tensor(0.0).to(loss1.device)
        if torch.isnan(loss2).any():
            loss2 = torch.tensor(0.0).to(loss2.device)
            
        loss = (loss1 + loss2) / pred1.numel()
        return loss
    
    def time_forward(self, pred1, pred2, targets):
        with torch.no_grad():
            compare1 = torch.abs(pred1 - targets)
            compare2 = torch.abs(pred2 - targets)
            mask = compare1 > compare2
        
        power = 2 if self.args.alio_time_loss == 'mse' else 1
        
        loss1 = torch.sum(torch.abs(pred1[mask] - pred2[mask].detach()).pow(power))
        loss2 = torch.sum(torch.abs(pred2[~mask] - pred1[~mask].detach()).pow(power))
        
        if torch.isnan(loss1).any():
            loss1 = torch.tensor(0.0).to(loss1.device)
        if torch.isnan(loss2).any():
            loss2 = torch.tensor(0.0).to(loss2.device)
            
        loss = (loss1 + loss2) / pred1.numel()
        return loss
    
    def forward(self, preds, targets):
        assert len(preds.size()) == 4, f"The shape of pred should be (batch_size, num_samples, pred_len, num_features), but got {preds.size()}"
        _, num_samples, _, _ = preds.size()
        assert num_samples == self.num_samples
        
        count = 0
        time_loss = torch.tensor(0.0).to(preds.device)
        freq_loss = torch.tensor(0.0).to(preds.device)
        for sample_idx in range(num_samples - 1):
            for sample_idx2 in range(sample_idx + 1, num_samples):
                if sample_idx == sample_idx2:
                    continue
                gap = abs(sample_idx - sample_idx2) * self.lag

                pred1 = preds[:, sample_idx, gap:, :]
                pred2 = preds[:, sample_idx2, :-gap, :]
                target = targets[:, sample_idx, gap:, :]
                
                if self.time_weight > 0:
                    time_loss += self.time_forward(pred1, pred2, target)
                if self.freq_weight > 0:
                    freq_loss += self.freq_forward(pred1, pred2, target)
                
                count += 1
                self.step += 1
        time_loss /= count
        freq_loss /= count
        return time_loss, freq_loss