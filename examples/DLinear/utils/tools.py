import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import matplotlib

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.figure(dpi=600)
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

    
def visual_multi_step(true1, true2, preds1=None, preds2=None, name='./pic/test.pdf', step=1):
    true = np.concatenate((true1, true2[-1:]))
    preds2 = np.concatenate((preds1[:0], preds2))
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.figure(dpi=600)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.plot(preds1, label='Prediction1', linewidth=2)
    plt.plot(preds2, label='Prediction2', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    
def visual_overlap(true, preds, name='./pic/test.pdf', num_samples=5, lag=1):
    assert num_samples <= 10
    assert num_samples >= 2
    try:
        colormap = {
            0: 'red',
            1: 'orange',
            2: 'green',
            3: 'blue',
            4: 'purple',
            5: 'brown',
            6: 'pink',
            7: 'gray',
            8: 'olive',
            9: 'cyan'
        }
        
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.figure(dpi=600)
        print(true.shape, preds.shape, num_samples, true, preds, file=open('test.txt', 'w'))
        true = np.concatenate((true[0, :], true[num_samples, -num_samples:]))
        plt.plot(true, label='GroundTruth', linewidth=1, color='black', linestyle='--')
        
        consistency_mae = 0
        consistency_mse = 0
        for i in range(num_samples - 1):
            pred1 = preds[lag * i, lag:]
            pred2 = preds[lag * (i + 1), :-lag]
            consistency_mae += np.mean(np.abs(pred1 - pred2))
            consistency_mse += np.mean((pred1 - pred2) ** 2)
        consistency_mae = -np.log(consistency_mae / (num_samples - 1))
        consistency_mse = -np.log(consistency_mse / (num_samples - 1))
        
        plt.title(f'Consistency MAE: {consistency_mae:.4f}, Consistency MSE: {consistency_mse:.4f}')
        
        for i in range(num_samples):
            x_range = range(lag * i, lag * i + len(preds[lag * i, :]))
            y = preds[lag * i, :]
            plt.plot(x_range, y, label=f'Prediction {i}', linewidth=1, color=colormap[i])
        plt.legend()
        plt.savefig(name, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)
        print('Error in visualization')
        pass

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))