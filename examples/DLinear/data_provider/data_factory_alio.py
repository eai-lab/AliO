from data_provider.data_loader_alio import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_Custom2
from torch.utils.data import DataLoader
import torch
import sys

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'custom2': Dataset_Custom2,
}

class FastDataLoader:
    def __init__(self, dataset, dataloader, device="cuda"):
        self.dataset = dataset
        self.dataloader = dataloader
        
        self.data_x = torch.tensor(self.dataset.data_x).to(device)
        self.data_y = torch.tensor(self.dataset.data_y).to(device)
        
        self.lag = self.dataset.lag
        self.seq_len = self.dataset.seq_len
        self.label_len = self.dataset.label_len
        self.pred_len = self.dataset.pred_len
        self.num_samples = self.dataset.num_samples
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for idx in self.dataloader:
            seq_x_batch = []
            seq_y_batch = []
            seq_x_mark_batch = []
            seq_y_mark_batch = []
            for index in idx:
                if self.num_samples == None:
                    s_begin = index
                    s_end = s_begin + self.seq_len
                    r_begin = s_end - self.label_len
                    r_end = r_begin + self.label_len + self.pred_len

                    seq_x = self.data_x[s_begin:s_end]
                    seq_y = self.data_y[r_begin:r_end]
                    seq_x_mark = torch.zeros((seq_x.shape[0], 1))
                    seq_y_mark = torch.zeros((seq_x.shape[0], 1))

                    seq_x_batch.append(seq_x)
                    seq_y_batch.append(seq_y)
                    seq_x_mark_batch.append(seq_x_mark)
                    seq_y_mark_batch.append(seq_y_mark)
                else:
                    seq_xs = []
                    seq_ys = []
                    seq_x_marks = []
                    seq_y_marks = []
                    for i in range(self.num_samples):
                        s_begin = index + i * self.lag
                        s_end = s_begin + self.seq_len
                        r_begin = s_end - self.label_len
                        r_end = r_begin + self.label_len + self.pred_len

                        seq_x = self.data_x[s_begin:s_end]
                        seq_y = self.data_y[r_begin:r_end]
                        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
                        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

                        seq_xs.append(seq_x)
                        seq_ys.append(seq_y)
                        seq_x_marks.append(seq_x_mark)
                        seq_y_marks.append(seq_y_mark)
                        
                    seq_xs = torch.stack(seq_xs, dim=0)   # (num_samples, seq_len, num_features)
                    seq_ys = torch.stack(seq_ys, dim=0)
                    seq_x_marks = torch.stack(seq_x_marks)
                    seq_y_marks = torch.stack(seq_y_marks)
                    
                    seq_x_batch.append(seq_xs)
                    seq_y_batch.append(seq_ys)
                    seq_x_mark_batch.append(seq_x_marks)
                    seq_y_mark_batch.append(seq_y_marks)
            
            seq_x_batch = torch.stack(seq_x_batch, dim=0)  
            seq_y_batch = torch.stack(seq_y_batch, dim=0)
            seq_x_mark_batch = torch.stack(seq_x_mark_batch, dim=0)
            seq_y_mark_batch = torch.stack(seq_y_mark_batch, dim=0)
            
            yield (seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch)

def data_provider(args, flag, lag=1):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.alio_test_num_samples * 2  # bsz=1 for evaluation
        freq = args.freq
        num_samples = None
        lag = 1
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
        num_samples = None
        lag = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
        num_samples = args.alio_num_samples
        lag = lag

    if not hasattr(args, 'fast_dataloader'):
        setattr(args, 'fast_dataloader', False)

    if args.data == 'custom' and args.fast_dataloader:
        Data = Dataset_Custom2
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        num_samples=num_samples,
        lag=lag,
    )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        persistent_workers=True,
        pin_memory=True, prefetch_factor=16)

    if args.data == 'custom' and args.fast_dataloader:
        print(f">> {flag}: Enable Fast dataloader")
        print(f">> {flag}: Enable Fast dataloader", file=sys.stderr)
        data_loader = FastDataLoader(data_set, data_loader)    

    return data_set, data_loader
