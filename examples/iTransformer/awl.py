import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, verbose=False, print_interval=100, args=None):
        super(AutomaticWeightedLoss, self).__init__()
        
        self.args = args
        
        print(f">> AWL: num={num}, verbose={verbose}, print_interval={print_interval}")
        
        params = torch.ones(num, requires_grad=True)        
        self.params = torch.nn.Parameter(params)
        self.steps = 0
        self.verbose = verbose
        self.print_interval = print_interval

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            
        if self.training:
            if self.verbose and self.steps % self.print_interval == 0:
                print(f"params: {self.params}")
                with torch.no_grad():
                    print(f"coeff: {0.5 / (self.params ** 2)}")
            self.steps += 1
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())