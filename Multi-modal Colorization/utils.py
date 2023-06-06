import torch

def rgb2xyz(rgb):  # rgb from [0,1]
    # array([[0.412453, 0.357580, 0.180423],
    #        [0.212671, 0.715160, 0.072169],
    #        [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    return out


def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)

    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc

    return out


def rgb2lab(rgb, l_cent=50, l_norm=100, ab_norm=110):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - l_cent) / l_norm
    ab_rs = lab[:, 1:, :, :] / ab_norm
    out = torch.cat((l_rs, ab_rs), dim=1)

    return out


def lab2rgb(lab_rs, l_cent=50, l_norm=100, ab_norm=110):
    l = lab_rs[:, [0], :, :] * l_norm + l_cent
    ab = lab_rs[:, 1:, :, :] * ab_norm
    lab = torch.cat((l, ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))

    return out


# lightness
from skimage import color
import warnings
import numpy as np

def rgb2lab_1d(in_rgb):
    # take 1d numpy array and do color conversion
    # print('in_rgb', in_rgb)
    return color.rgb2lab(in_rgb[np.newaxis, np.newaxis, :]).flatten()

def lab2rgb_1d(in_lab, clip=True, dtype='uint8'):
    warnings.filterwarnings("ignore")
    tmp_rgb = color.lab2rgb(in_lab[np.newaxis, np.newaxis, :]).flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    if dtype == 'uint8':
        tmp_rgb = np.round(tmp_rgb * 255).astype('uint8')
    return tmp_rgb

def snap_ab(input_l, input_rgb, return_type='rgb'):
    ''' given an input lightness and rgb, snap the color into a region where l,a,b is in-gamut
    '''
    T = 20
    warnings.filterwarnings("ignore")
    input_lab = rgb2lab_1d(np.array(input_rgb))  # convert input to lab
#     input_lab = rgb2lab_1d(input_rgb) # convert input to lab

    conv_lab = input_lab.copy()  # keep ab from input
    for t in range(T):
        conv_lab[0] = input_l  # overwrite input l with input ab
        old_lab = conv_lab
        tmp_rgb = color.lab2rgb(conv_lab[np.newaxis, np.newaxis, :]).flatten()
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
        conv_lab = color.rgb2lab(tmp_rgb[np.newaxis, np.newaxis, :]).flatten()
        dif_lab = np.sum(np.abs(conv_lab - old_lab))
        if dif_lab < 1:
            break
        # print(conv_lab)

    conv_rgb_ingamut = lab2rgb_1d(conv_lab, clip=True, dtype='uint8')
    if (return_type == 'rgb'):
        return conv_rgb_ingamut

    elif(return_type == 'lab'):
        conv_lab_ingamut = rgb2lab_1d(conv_rgb_ingamut)
        return conv_lab_ingamut
