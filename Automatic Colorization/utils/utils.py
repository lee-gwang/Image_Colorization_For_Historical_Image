from turtle import pd
import pandas as pd
import numpy as np
import random
import torch
import os
# from warmup_scheduler import GradualWarmupScheduler

def load_data(CFG):
    dataset = CFG.dataset_path
    try:
        df = pd.read_csv(f'{dataset}')
        print('load preprocess 5fold csv')
        return df
    
    except:
        print('')        
        return df


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


# else
def init_logger(log_file, name):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(name)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

# class GradualWarmupSchedulerV2(GradualWarmupScheduler):
#     def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
#         super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
#     def get_lr(self):
#         if self.last_epoch > self.total_epoch:
#             if self.after_scheduler:
#                 if not self.finished:
#                     self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
#                     self.finished = True
#                 return self.after_scheduler.get_lr()
#             return [base_lr * self.multiplier for base_lr in self.base_lrs]
#         if self.multiplier == 1.0:
#             return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
#         else:
#             return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
def lr_visualize(CFG):
    import matplotlib.pyplot as plt
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler== 'warmupv2':
            scheduler_cosine=CosineAnnealingWarmRestarts(optimizer, CFG.cosine_epo)
            scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=CFG.warmup_factor, total_epoch=CFG.warmup_epo, after_scheduler=scheduler_cosine)
            scheduler=scheduler_warmup 
        return scheduler

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = optim.Adam(model, CFG.lr)

    # scheduler_warmup is chained with schduler_steplr
    scheduler = get_scheduler(optimizer)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    lrs = []
    for epoch in range(CFG.epochs):
        lrs.append(optimizer.param_groups[0]['lr'])

        optimizer.step() 
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        else:
            scheduler.step()
    #     scheduler.step(epoch)
    
    plt.plot(lrs)
    plt.show()
    print(f'last lr {lrs[-1] : .6f}')


# bits
def convert_bits(x, n_bits_in=8, n_bits_out=8):
    """Quantize / dequantize from n_bits_in to n_bits_out."""
    x = torch.tensor(x)
    if n_bits_in == n_bits_out:
        return x
    x = x.type(torch.float32)
    x = x / 2**(n_bits_in - n_bits_out)
    x = x.type(torch.int32)
    x = x.numpy()
    return x

def labels_to_bins(labels, num_symbols_per_channel = 8):
    """Maps each (R, G, B) channel triplet to a unique bin.
    Args:
    labels: 4-D Tensor, shape=(batch_size, H, W, 3).
    -->
    labels: 4-D Tensor, shape=(batch_size, 3, H, W).

    num_symbols_per_channel: number of symbols per channel.
    Returns:
    labels: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
    """
    labels = labels.type(torch.float32)
    channel_hash = [num_symbols_per_channel**2, num_symbols_per_channel, 1.0]
    channel_hash = torch.tensor(channel_hash)
    labels = labels * channel_hash

    labels = torch.sum(labels, axis=1) # axis = -1
    labels = labels.type(torch.int32)
    return labels


def bins_to_labels(bins, num_symbols_per_channel=8):
    """Maps back from each bin to the (R, G, B) channel triplet.
    Args:
    bins: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
    num_symbols_per_channel: number of symbols per channel.
    Returns:
    labels: 4-D Tensor, shape=(batch_size, H, W, 3)
    """
    labels = []
    factor = int(num_symbols_per_channel**2)

    for _ in range(3):
        #channel = tf.math.floordiv(bins, factor)
        channel = torch.div(bins, factor, rounding_mode='floor')
        labels.append(channel)

#         bins = tf.math.floormod(bins, factor)
        bins = torch.remainder(bins, factor)
        factor = factor // num_symbols_per_channel
    return torch.stack(labels, axis=3)

from skimage import color

def lab2rgb(L, ab):
    """
    L : range: [-1, 1], torch tensor
    ab : range: [-1, 1], torch tensor

    input : torch tensor, (bs, c, h, w) --> 4D tensor
    output : numpy array
    """
    ab2 = ab * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = torch.cat([L2, ab2], dim=1)
    Lab = Lab.detach().cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float32), (0, 2, 3, 1))
    rgb = color.lab2rgb(Lab) * 255
    return rgb

def lab_norm(rgb, l_cent=50, l_norm=100, ab_norm=110):
    """
    input : torch tensor, (bs, c, h, w) --> 4D tensor
    output : torch tensor 
    """
    l_rs = (rgb[:, [0], :, :] / 50) - 1 # [-1, 1]
    ab_rs = rgb[:, 1:, :, :] / 110
    # out = torch.cat((l_rs, ab_rs), dim=1)

    return l_rs, ab_rs
