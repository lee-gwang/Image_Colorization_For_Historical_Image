import datetime
import glob
import os
import sys

import cv2
import numpy as np
import torch
from einops import rearrange
from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget
from skimage import color

from .lab_gamut import snap_ab
from .ui_control import UIControl
import torch.nn as nn


def patch_inference(gray_imgs, model, img_size=768, stride=740, scale=3, batch_size=32, device='cpu'):
    ensemble_img = []
    len_weights = 0 # total of the ensemble weights
    p_w = 1 # patch weight
    i_w = 1 # image weight
    stride = 740
    img_size = 768
    scale = 3
    img_size2 = int(img_size * scale)
    stride2   = int(stride * scale)
    _,c,h,w = gray_imgs.shape
    ##############
    crop = []
    position = []
    batch_count = 0
    batch_size = 32
    device = 'cpu'

    result_img = np.zeros([3, h, w])
    voting_mask = np.zeros([3, h, w])
    for top in range(0, h, stride2):
        for left in range(0, w, stride2):#-img_size+stride
            piece = torch.zeros([1, 1, img_size2, img_size2])
            temp = gray_imgs[:, :, top:top+img_size2, left:left+img_size2] # bs, c, h, w

            piece[:, :, :temp.shape[2], :temp.shape[3]] = temp
            # print('piece2 : ',piece.shape)

            # crop.append(piece)
            crop.append(piece)
            position.append([top, left])
            batch_count += 1
            if batch_count == batch_size:
                crop = torch.cat(crop, axis=0).to(device)
                pred  = model(crop)
                #
                pred  = model(nn.Upsample(scale_factor=1/scale, mode='bilinear')(crop)) # downsampling
                pred = nn.Upsample(scale_factor=scale, mode='bilinear')(pred)

                # # area
                # pred = model(F.interpolate(crop, size=img_size, mode='area'))
                # pred = F.interpolate(pred, size=img_size2, mode='area')


                #

                pred = torch.cat([crop, pred], 1)
                pred = pred.cpu().detach().numpy()
                #pred = model(crop)*255
                #pred = pred.detach().cpu().numpy()
                crop = []
                batch_count = 0
                for num, (t, l) in enumerate(position):
                    piece = pred[num]
                    c_, h_, w_ = result_img[:, t:t+img_size2, l:l+img_size2].shape
                    result_img[:, t:t+img_size2, l:l+img_size2] += piece[:, :h_, :w_]
                    voting_mask[:, t:t+img_size2, l:l+img_size2] += 1
                position = []
    if batch_count != 0: # batch size만큼 안채워지면
        # crop = torch.from_numpy(np.array(crop)).permute(0,3,1,2).to(device)
        # crop = torch.stack(crop, axis=0).to(device)
        crop = torch.cat(crop, axis=0).to(device)

        pred  = model(nn.Upsample(scale_factor=1/scale, mode='bilinear')(crop))
        pred = nn.Upsample(scale_factor=scale, mode='bilinear')(pred)
        pred = torch.cat([crop,pred], 1)
        pred = pred.cpu().detach().numpy()
        crop = []
        batch_count = 0
        for num, (t, l) in enumerate(position):
            piece = pred[num]
            c_, h_, w_ = result_img[:, t:t+img_size2, l:l+img_size2].shape
            result_img[:, t:t+h_, l:l+w_] += piece[:, :h_, :w_]
            voting_mask[:, t:t+h_, l:l+w_] += 1
        position = []


    result_img = result_img/voting_mask
    ensemble_img.append(result_img*p_w)
    len_weights += p_w

    #########
    # image-wise ensmeble (나중에는, for구문 안에 넣어야할듯,,)
    gray_imgs = gray_imgs.to(device)
    #
    pred = model(nn.Upsample(size=img_size, mode='bilinear')(gray_imgs))
    pred = nn.Upsample(size=(h,w), mode='bilinear')(pred)

    # pred = model(F.interpolate(gray_imgs, size=img_size, mode='area'))
    # pred = F.interpolate(pred, size=(h,w), mode='area')
    #
    pred = torch.cat([gray_imgs, pred], 1).cpu().detach().numpy() # bs, c, h, w
    ensemble_img.append(pred[0]*i_w)
    len_weights += i_w



    # ensmeble
    # result_img = np.sum(ensemble_img, axis=0)/len(ensemble_img)
    result_img = np.sum(ensemble_img, axis=0)/len_weights

    # ab, lab
    return np.around(result_img*255).astype(np.uint8).transpose(1,2,0)


class GUIDraw(QWidget):

    # Signals
    update_color = pyqtSignal(str)
    update_gammut = pyqtSignal(object)
    used_colors = pyqtSignal(object)
    update_ab = pyqtSignal(object)
    update_result = pyqtSignal(object)

    def __init__(self, model=None, nohint_model=None, load_size=224, win_size=512, device='cpu', patch=False):
        QWidget.__init__(self)
        self.image_file = None
        self.pos = None
        self.model = model
        # add
        self.nohint_model = nohint_model

        self.win_size = win_size
        self.load_size = load_size
        self.device = device
        self.setFixedSize(win_size, win_size)
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.move(win_size, win_size)
        self.movie = True
        self.init_color()  # initialize color
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'   # stroke or point
        self.image_loaded = False
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0

        # patch inference
        self.patch = patch

        
    def clock_count(self):
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file):
        self.read_image(image_file)
        # patch-wise inference
        if self.patch:
            im_full = self.im_full
            gray = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
            gray = np.stack([gray, gray, gray], -1)
            l_img = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)[:,:,[0]].transpose((2,0,1))
            l_img = torch.from_numpy(l_img).type(torch.FloatTensor).to(self.device)/255
            lab = patch_inference(l_img.unsqueeze(0), self.nohint_model)
            self.my_results = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # 왜.. gbr밖에
            ab = lab[:,:,[1,2]]


        # image-wise inference
        else:
            # my model
            im_full = cv2.resize(self.im_full, (768,768), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
            gray = np.stack([gray, gray, gray], -1)
            l_img = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)[:,:,[0]].transpose((2,0,1))
            l_img = torch.from_numpy(l_img).type(torch.FloatTensor).to(self.device)/255
            ab = self.nohint_model(l_img.unsqueeze(0))[0]#.detach().cpu().numpy().transpose((1,2,0))

            lab = torch.cat([l_img, ab], axis=0).permute(1,2,0).cpu().detach().numpy() * 255 # h,w,c
            lab = lab.astype(np.uint8)
            self.my_results = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # 왜.. gbr밖에
            ab = ab.permute(1,2,0).cpu().detach().numpy() * 255
            


        #######
        # 저장용
        # ab = ab.permute(1,2,0).cpu().detach().numpy() * 255
        ab = cv2.resize(ab, (self.im_full.shape[1],self.im_full.shape[0]), interpolation=cv2.INTER_AREA) # INTER_CUBIC
        im_l = cv2.cvtColor(self.im_full, cv2.COLOR_BGR2LAB)[:,:,[0]]
        org_my_results = np.concatenate([im_l, ab], axis=2)
        org_my_results = org_my_results.astype(np.uint8)

        self.org_my_results = cv2.cvtColor(org_my_results, cv2.COLOR_LAB2BGR) # 왜.. gbr밖에

        ##############################
        #
        self.reset()

    def get_batches(self, img_dir):
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file):
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_AREA) # INTER_CUBIC

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_AREA) #INTER_CUBIC
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA) # INTER_CUBIC
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.org_im_l = color.rgb2lab(self.im_full[:, :, ::-1])[:, :, [0]]
        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

    def update_im(self):
        self.update()
        QApplication.processEvents()

    def update_ui(self, move_point=True):
        if self.ui_mode == 'none':
            return False
        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))

        if self.ui_mode == 'point':
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
                    # self.predict_color()

        if self.ui_mode == 'stroke':
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        if self.ui_mode == 'erase':
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
                # self.predict_color()
        return is_predict

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.compute_result()
        self.update()

    def scale_point(self, pnt):
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QPoint(x, y)
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt.x(), pnt.y()))
                return None

    def init_color(self):
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            # self.emit(SIGNAL('update_gamut'), L)
            self.update_gammut.emit(L)

            used_colors = self.uiControl.used_colors()
            # self.emit(SIGNAL('used_colors'), used_colors)
            self.used_colors.emit(used_colors)

            snap_color = self.calibrate_color(self.user_color, pos)
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)
            # self.emit(SIGNAL('update_ab'), c)
            self.update_ab.emit(c)

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L = self.im_l[y, x]
        snap_color = snap_ab(mean_L, color_array)
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def set_color(self, c_rgb):
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()

    def erase(self):
        self.eraseMode = not self.eraseMode

    def load_image(self):
        img_path = QFileDialog.getOpenFileName(self, 'load an input image')[0]
        if img_path is not None and os.path.exists(img_path):
            self.init_result(img_path)

    def save_result(self):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)
        print(path)
        # add
        #########
        # original size image
        #########
        org_ab = cv2.resize(self.ab, (self.im_full.shape[1],self.im_full.shape[0]), interpolation=cv2.INTER_AREA) # INTER_CUBIC
        org_ab = org_ab * 110
        org_pred_lab = np.concatenate((self.org_im_l, org_ab), axis=2)
        org_pred_lab = (np.clip(color.lab2rgb(org_pred_lab), 0, 1) * 255.)

        saved_rgb = self.org_my_results * 0.5 + org_pred_lab * 0.5
        # saved_rgb = self.org_my_results

        self.result = saved_rgb.astype('uint8')

        #
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # save_path = "_".join([path, suffix])
        save_path = os.path.join('/'.join(path.split('/')[:-1]), 'output_img')

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        # cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        # cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(save_path, f'{path.split("/")[-1]}.png'), result_bgr)


    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()

    def compute_result(self):
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1)) # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))#(3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2,0,1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :]-50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255-self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                        h=self.load_size//self.model.patch_size, w=self.load_size//self.model.patch_size,
                        p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()
        self.ab = ab

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA) # INTER_CUBIC
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        #########
        # my model
        #########
        my_results = cv2.resize(self.my_results, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255.)
        # pred_rgb = my_results
        pred_rgb = my_results*0.5 + pred_rgb*0.5

        pred_rgb = pred_rgb.astype('uint8')


        #####################################################
        self.result = pred_rgb
        # self.emit(SIGNAL('update_result'), self.result)
        self.update_result.emit(self.result)
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        painter.setRenderHint(QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            qImg = QImage(im.tostring(), im.shape[1], im.shape[0], QImage.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)

        self.uiControl.update_painter(painter)
        painter.end()

    # def wheelEvent(self, event):
    #     d = event.delta() / 120
    #     self.brushWidth = min(4.05 * self.scale, max(0, self.brushWidth + d * self.scale))
    #     print('update brushWidth = %f' % self.brushWidth)
    #     self.update_ui(move_point=True)
    #     self.update()

    def is_same_point(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        dx = pos1.x() - pos2.x()
        dy = pos1.y() - pos2.y()
        d = dx * dx + dy * dy
        # print('distance between points = %f' % d)
        return d < 25

    def mousePressEvent(self, event):
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self.compute_result()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                self.compute_result()

    def mouseMoveEvent(self, event):
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def mouseReleaseEvent(self, event):
        pass

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)  # 28 * 8
