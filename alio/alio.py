from typing import Tuple
import torch
import torch.nn as nn

class AliO(nn.Module):
    def __init__(
        self,
        num_samples: int=None,
        lag: int=1, 
        time_loss: str='mse',
        freq_loss: str='mse',
        **kwargs
    ):
        """
        This is the official implementation of AliO.

        Args:
            num_samples (int): The number of sequential samples (N).
            lag (int): The lag between two samples. Defaults to 1.
            time_loss (str): The loss function in the time domain. Defaults to 'mse'.
            freq_loss (str): The loss function in the frequency domain. Defaults to 'mse'.
            kwargs: Additional arguments. It does not have any effect.
        """
        super(AliO, self).__init__()

        # Check the input arguments
        assert num_samples > 1, f"num_samples should be greater than 1, but got {num_samples}"
        assert lag > 0, f"lag should be greater than 0, but got {lag}"
        
        self.num_samples = num_samples
        self.lag = lag
        self.time_loss = time_loss
        self.freq_loss = freq_loss
    
    def loss_fn(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculate the loss function in the time domain.
        It does not have mean operation, so you should calculate the mean loss in the caller function

        Args:
            x (torch.Tensor): data1
            y (torch.Tensor): data2

        Raises:
            ValueError: Unsupported loss operator

        Returns:
            _type_: loss
        """
        if self.time_loss == 'mse':
            return torch.sum((x - y).pow(2))
        elif self.time_loss == 'mae':
            return torch.sum(torch.abs(x - y))
        elif self.time_loss == 'huber':
            return torch.sum(nn.SmoothL1Loss(x, y))
        
        raise ValueError(f"Invalid time_loss: {self.time_loss}")
    
    def freq_forward(self, pred1: torch.Tensor, pred2: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the frequency loss.
        
        Args:
            pred1 (torch.Tensor): The prediction of the first sample. (batch_size, pred_len, num_features)
            pred2 (torch.Tensor): The prediction of the second sample.  (batch_size, pred_len, num_features)
            targets (torch.Tensor): The target data.    (batch_size, pred_len, num_features)
        """
        # Apply the FFT to the prediction and target
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
        loss1 = self.loss_fn(pred1[mask], pred2[mask].detach())
        loss2 = self.loss_fn(pred2[~mask], pred1[~mask].detach())
        
        # Handle the NaN value. NaN is casued by the len(loss1) == 0 or len(loss2) == 0, which means one of pred1 and pred2 has bias.
        if torch.isnan(loss1).any():
            loss1 = torch.tensor(0.0).to(loss1.device)
        if torch.isnan(loss2).any():
            loss2 = torch.tensor(0.0).to(loss2.device)
        
        # Calculate the average loss. -> Average the loss in the frequency domain
        loss = (loss1 + loss2) / pred1.numel()
        return loss
    
    def time_forward(self, pred1: torch.Tensor, pred2: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the time loss.

        Args:
            pred1 (_type_): The prediction of the first sample.
            pred2 (_type_): The prediction of the second sample.
            targets (_type_): The target data.

        Returns:
            _type_: (time_loss, freq_loss)
        """
        # Calculate the absolute difference between the prediction and target to obtain the mask
        with torch.no_grad():
            compare1 = torch.abs(pred1 - targets)
            compare2 = torch.abs(pred2 - targets)
            mask = compare1 > compare2
        
        # Calculate the loss
        loss1 = self.loss_fn(pred1[mask], pred2[mask].detach())
        loss2 = self.loss_fn(pred2[~mask], pred1[~mask].detach())
        
        # Handle the NaN value. NaN is casued by the len(loss1) == 0 or len(loss2) == 0, which means one of pred1 and pred2 has bias.
        if torch.isnan(loss1).any():
            loss1 = torch.tensor(0.0).to(loss1.device)
        if torch.isnan(loss2).any():
            loss2 = torch.tensor(0.0).to(loss2.device)
        
        # Calculate the average loss. -> Average the loss in the time domain
        loss = (loss1 + loss2) / pred1.numel()
        return loss
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the AliO loss function.
        The preds and targets should have the same shape (batch_size, num_samples, pred_len, num_features).
        """
        # preds: (batch_size, num_samples, pred_len, num_features)
        # targets: (batch_size, num_samples, pred_len, num_features)
        assert len(preds.size()) == 4, f"The shape of pred should be (batch_size, num_samples, pred_len, num_features), but got {preds.size()}"
        assert preds.size() == targets.size(), f"The shape of pred and target should be same, but got {preds.size()} and {targets.size()}"
        
        _, num_samples, _, _ = preds.size()
        # Check the number of samples
        assert num_samples == self.num_samples
        
        # Initialize the loss and count
        count = 0
        time_loss = torch.tensor(0.0).to(preds.device)
        freq_loss = torch.tensor(0.0).to(preds.device)
        
        # Compare the all pairs of samples
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
                time_loss += self.time_forward(pred1, pred2, target)
                freq_loss += self.freq_forward(pred1, pred2, target)
                
                count += 1
        
        # Calculate the average loss
        time_loss /= count
        freq_loss /= count
        
        return time_loss, freq_loss