"""
hint + nohint model(mine)
"""

import sys
import os
import argparse

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from timm.models import create_model
import torch

from gui import gui_main
import modeling

import segmentation_models_pytorch as smp
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser('Colorization UI', add_help=False)
    # Directories
    parser.add_argument('--model_path', type=str, default='path/to/checkpoints', help='checkpoint path of model')
    parser.add_argument('--target_image', default='path/to/image', type=str, help='validation dataset path')
    parser.add_argument('--device', default='cpu', help='device to use for testing')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')
    parser.add_argument('--patch', action='store_true', help='patch wise inference')

    args = parser.parse_args()

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
    )

    return model

# 모델 저장할때, nn.module로 한번 감싸고, dataparra을 썼더니, 모델이름이 드럽네

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()

    model = get_model(args)
    model.to(args.device)
    checkpoint = torch.load('./saved_models/best/icolorit_base_4ch_patch16_224.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # no hint model
    print('Creating model:', 'nohint model' )
    nohint_model = smp.Unet(
            encoder_name='efficientnet-b1',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,        # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )
    # 쓸모없는 키 이름 지우기
    from collections import OrderedDict
    temp_dict = OrderedDict()
    weight = torch.load('./pretrained/fold0_best_e49.pth', map_location='cpu')
    for k, w in weight.items():
        temp_dict[k.replace('module.model.','')] = w

    #
    nohint_model.load_state_dict(temp_dict)
    nohint_model.eval()
    #
    app = QApplication(sys.argv)
    ex = gui_main.IColoriTUI(color_model=model, nohint_model=nohint_model, img_file=args.target_image,
                             load_size=args.input_size, win_size=1280, device=args.device, patch=args.patch)
    ex.setWindowIcon(QIcon('gui/icon.png'))
    ex.setWindowTitle('iColoriT')
    ex.show()
    sys.exit(app.exec_())
# python ./iColoriT_demo/icolorit_ui2.py --model_path ./saved_models/best/icolorit_base_4ch_patch16_224.pth --target_image /home/data/colorization/SR_magic/wYkp8TWqcrLk8y8N-lv7xg_output_spnv_x2.png