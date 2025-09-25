import torch
import torch.nn as nn
import torch.nn.functional as F

from awl import AutomaticWeightedLoss

class AliO(nn.Module):
    def __init__(self, num_samples, lag=1, args=None, **kwargs):
        """_summary_

        Args:
            num_samples (int): The number of sequential samples to be closed.
            lag (int, optional): The lag between two samples. Defaults to 1.
            args (optional): Arguments. Defaults to None.
                ali_time_weight (float): The weight of time domain loss. Defaults to 1.0.
                ali_freq_weight (float): The weight of frequency domain loss. Defaults to 1.0.
                ali_time_loss (str): The loss function in time domain. Defaults to "mse".
                ali_freq_loss (str): The loss function in frequency domain. Defaults to "mse".
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
        # Calculate the FFT of the prediction and target
        pred1 = torch.fft.fft(pred1, dim=-2, norm='forward')
        pred2 = torch.fft.fft(pred2, dim=-2, norm='forward')
        targets = torch.fft.fft(targets, dim=-2, norm='forward')
        
        # Calculate the absolute difference between the prediction and target to obtain the mask
        with torch.no_grad():
            compare1 = torch.abs(pred1 - targets)
            compare2 = torch.abs(pred2 - targets)
            mask = compare1 > compare2
        
        # Calculate the loss. You can use other distance function or divergence function.
        power = 2 if self.args.alio_freq_loss == 'mse' else 1
        
        # Calculate the loss
        loss1 = torch.sum(torch.abs(pred1[mask] - pred2[mask].detach()).pow(power))
        loss2 = torch.sum(torch.abs(pred2[~mask] - pred1[~mask].detach()).pow(power))
        
        # Handle the NaN value. NaN is casued by the len(loss1) == 0 or len(loss2) == 0, which means one of pred1 and pred2 has bias.
        if torch.isnan(loss1).any():
            loss1 = torch.tensor(0.0).to(loss1.device)
        if torch.isnan(loss2).any():
            loss2 = torch.tensor(0.0).to(loss2.device)
        
        # Calculate the average loss
        loss = (loss1 + loss2) / pred1.numel()
        return loss
    
    def time_forward(self, pred1, pred2, targets):
        # Calculate the absolute difference between the prediction and target to obtain the mask
        with torch.no_grad():
            compare1 = torch.abs(pred1 - targets)
            compare2 = torch.abs(pred2 - targets)
            mask = compare1 > compare2
        
        # Calculate the loss. You can use other distance function or divergence function.
        power = 2 if self.args.alio_time_loss == 'mse' else 1
        
        # Calculate the loss
        loss1 = torch.sum(torch.abs(pred1[mask] - pred2[mask].detach()).pow(power))
        loss2 = torch.sum(torch.abs(pred2[~mask] - pred1[~mask].detach()).pow(power))
        
        # Handle the NaN value. NaN is casued by the len(loss1) == 0 or len(loss2) == 0, which means one of pred1 and pred2 has bias.
        if torch.isnan(loss1).any():
            loss1 = torch.tensor(0.0).to(loss1.device)
        if torch.isnan(loss2).any():
            loss2 = torch.tensor(0.0).to(loss2.device)
        
        # Calculate the average loss
        loss = (loss1 + loss2) / pred1.numel()
        return loss
    
    def forward(self, preds, targets):
        """_summary_
            targets and preds have lagged time data in num_samples axis.
            For example, targets[:, 1:, :, :] is equal to targets[:, :-1, :, :].
        """
        # preds: (batch_size, num_samples, pred_len, num_features)
        # targets: (batch_size, num_samples, pred_len, num_features)
        assert len(preds.size()) == 4, f"The shape of pred should be (batch_size, num_samples, pred_len, num_features), but got {preds.size()}"
        _, num_samples, _, _ = preds.size()
        assert num_samples == self.num_samples
        
        count = 0
        time_loss = torch.tensor(0.0).to(preds.device)
        freq_loss = torch.tensor(0.0).to(preds.device)
        
        # Compare the all combinations of samples
        for sample_idx in range(num_samples - 1):
            for sample_idx2 in range(sample_idx + 1, num_samples):
                if sample_idx == sample_idx2:
                    continue
                
                # Calculate the gap between two samples
                gap = abs(sample_idx - sample_idx2) * self.lag

                # Obtain the data in common timestamps between two samples
                pred1 = preds[:, sample_idx, gap:, :]
                pred2 = preds[:, sample_idx2, :-gap, :]
                target = targets[:, sample_idx, gap:, :]
                
                # Calculate the loss
                if self.time_weight > 0:
                    time_loss += self.time_forward(pred1, pred2, target)
                if self.freq_weight > 0:
                    freq_loss += self.freq_forward(pred1, pred2, target)
                
                count += 1
                self.step += 1
        
        # Calculate the average loss
        time_loss /= count
        freq_loss /= count
        return time_loss, freq_loss
