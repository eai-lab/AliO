from data_provider.data_loader_alio import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS
from data_provider.data_loader_alio_fast import Dataset_ETT_hour_Fast, DataLoader_ETT_hour_Fast, Dataset_Custom_Fast, DataLoader_Custom_Fast, Dataset_Solar_Fast, DataLoader_Solar_Fast
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS
}

data_dict_fast = {
    'ETTh1': Dataset_ETT_hour_Fast,
    'ETTh2': Dataset_ETT_hour_Fast,
    'custom': Dataset_Custom_Fast,
    'Solar': Dataset_Solar_Fast,
}

def data_provider(args, flag, lag=1):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
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
        batch_size = args.batch_size
        freq = args.freq
        num_samples = args.alio_num_samples
        lag = lag

    if not hasattr(args, 'fast_dataloader'):
        setattr(args, 'fast_dataloader', False)

    if args.fast_dataloader and args.data in data_dict_fast.keys():
        Data = data_dict_fast[args.data]

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle,
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
        pin_memory=True)
    
    if args.fast_dataloader and args.data in data_dict_fast.keys():
        print('Fast Dataloader')
        
        if args.data == 'custom':
            data_loader = DataLoader_Custom_Fast(data_set, data_loader)
        elif args.data == 'ETTh1' or args.data == 'ETTh2':
            data_loader = DataLoader_ETT_hour_Fast(data_set, data_loader)
        elif args.data == 'Solar':
            data_loader = DataLoader_Solar_Fast(data_set, data_loader)
        else:
            raise ValueError(f"Unsupported dataset for fast dataloader: {args.data}")
    
    return data_set, data_loader
