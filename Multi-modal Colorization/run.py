import os
from google.cloud import storage
import flask
import urllib.request
from PIL import Image, ImageOps
import numpy as np
from timm.models import create_model
import modeling
import torch
import torch.nn.functional as F

import datetime
import glob
import sys

import cv2
import numpy as np
from einops import rearrange
from skimage import color
import time
import pandas as pd
import datetime
import os
from utils import rgb2lab, lab2rgb, snap_ab

# network
from gfpgan.utils import GFPGANer
import clip

from collections import OrderedDict
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='디버그, gcp에 안올림')
args = parser.parse_args()

def test_infer(img_lq, model, scale, window_size, tile, tile_overlap):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################################
device = 'cpu'
# 사전에 json 파일을 다운로드 받을것!
# 역슬래쉬를 슬래쉬로 다 변경할 것
KEY_PATH = "Your JSON FILE PATH"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= KEY_PATH

# 미리 캐쉬해놓자..
# 서비스 계정 생성한 bucket 이름 입력
bucket_name = 'chat-ai-bot'    
storage_client = storage.Client()
buckets = list(storage_client.list_buckets())
bucket = storage_client.bucket(bucket_name)

def upload_img2gcp(source_file_name, destination_blob_name):
    """
    source_img = 경로포함/~.jpg
    dest_img = ~.jpg
    """
    
    # GCP에 업로드할 파일 절대경로, 경로 사이 역슬래쉬는 슬래쉬로 변환할 것
    source_file_name = source_file_name  
    # destination_blob_name : 업로드할 파일을 GCP에 저장할 때의 이름. 새로운 이미지를 넣을 때마다 바꾸어줘야 함

    # 이미지 업로드하기
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    upload_img_link = f'https://storage.googleapis.com/{bucket_name}/{destination_blob_name}'
    
    # blob.download_to_filename(source_file_name)
    
    return upload_img_link


class Draw:
    def __init__(self, model=None, image_file=None, device='cpu', color=False):
        self.model = model
        self.image_file = image_file
        self.device = device
        self.color = color
        
    def read_image(self, image_file):
        # self.result = None
        self.load_size = 224
        # self.image_file = image_file
        im_bgr = cv2.imread(image_file)
        im_full = im_bgr.copy()
        # get image for display
        h, w, c = im_full.shape
        self.h, self.w = h, w

        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA) # INTER_CUBIC
        self.im_bgr = im_bgr[:, :, ::-1]
        self.im_lab = color.rgb2lab(self.im_bgr)

        # lab 변환된 값은 다 0~1사이 값아님
        # color.rgb2lab 너무느림
        # im_full = color.rgb2lab(im_full[:, :, ::-1])
        # print(color.rgb2lab(im_full[:, :, ::-1])[:,:,1:].max())
        im_full = im_full[:, :, ::-1]
        im_full = torch.from_numpy(im_full.copy()).type(torch.FloatTensor)
        im_full = im_full.unsqueeze(0).permute(0, 3, 1, 2)
        im_full = rgb2lab(im_full).permute(0, 2, 3, 1).squeeze(0)

        # print(im_full[:,:,1:].max())
        self.org_size_l = im_full[:, :, 0]
    
    def compute_result(self, mask=None, idx=None, input_rgb=None, line_col=False, line_img_path=None):

        if mask is not None: # use the hint mask
            if line_col:
                mask, mask2 = mask[0], mask[1]

                # load line img
                line_img = cv2.imread(line_img_path)
                line_img = cv2.resize(line_img, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA) # INTER_CUBIC
                line_img = line_img[:, :, ::-1]

                # compute diff
                img_diff = cv2.absdiff(self.im_bgr, line_img).astype(np.float32) 
                img_diff = np.sum(img_diff, axis=2)

                # indexing
                # thres = np.quantile(img_diff, 0.999)
                temp = img_diff.reshape(-1)
                mink_value = temp[temp.argsort()[::-1][:100]].min()
                x = np.where(img_diff>=mink_value)[0]
                y = np.where(img_diff>=mink_value)[1]

                max_pos = [(y,x) for x,y in zip(x,y)]
                np.random.shuffle(max_pos)
                tls = max_pos[:10] # [(992, 806), (979, 813)]

                # colorize
                L = color.rgb2lab(line_img)[:,:,0]
                for tl in tls:
                    mask = cv2.rectangle(mask, tl, (tl[0]+1, tl[1]+1), tuple(snap_ab(L[(tl[1], tl[0])], line_img[(tl[1], tl[0])]).tolist()), -1)
                    mask2 = cv2.rectangle(mask2, tl, (tl[0]+1, tl[1]+1), 255, -1)

                im_mask0 = mask2 > 0.0
                self.im_mask0 = im_mask0.transpose((2, 0, 1))
                im_lab = color.rgb2lab(mask).transpose((2, 0, 1))

                self.im_ab0 = im_lab[1:3, :, :]


            else:
                mask, mask2 = mask[0], mask[1]
                tl, br = idx[0], idx[1] # (x,y)

                lightness = self.im_lab[:,:,0][tl[1],tl[0]] # L (h, w)

                color_ = tuple(snap_ab(lightness, input_rgb).tolist())

                mask = cv2.rectangle(mask, tl, br, color_, -1)
                mask2 = cv2.rectangle(mask2, tl, br, 255, -1)

                im_mask0 = mask2 > 0.0
                self.im_mask0 = im_mask0.transpose((2, 0, 1))
                im_lab = color.rgb2lab(mask).transpose((2, 0, 1))

                self.im_ab0 = im_lab[1:3, :, :]
        else: # not use the hint mask
            im = np.zeros((self.load_size, self.load_size, 3), np.uint8)
            mask = np.zeros((self.load_size, self.load_size, 1), np.uint8)
            
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
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device) # 3ch

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                        h=self.load_size//self.model.patch_size, w=self.load_size//self.model.patch_size,
                        p1=self.model.patch_size, p2=self.model.patch_size)[0]

        ab = ab.detach().numpy()
        # ab = ab.cpu().detach().numpy()

        ## gwang add
        if self.color:
            org_ab = self.im_lab[:,:,1:]/110
            # thres_ = 0.05
            # diff_mask = np.abs(ab-org_ab) > thres_ # 차이나는 부분은 mask=1으로 만들자
            # ab = ab*diff_mask
            # org_ab = org_ab*(1-diff_mask)
            # masked_org_ab = org_ab*diff_mask
            # ab = org_ab+(ab+masked_org_ab)/2

            # diff_mask = np.abs(ab-org_ab) > thres_ # 차이나는 부분은 mask=1으로 만들자
            # ab = ab*diff_mask + org_ab*(1-diff_mask)

            ab = ab * 0.7 +org_ab*0.3

        ab_win = cv2.resize(ab, (self.w, self.h), interpolation=cv2.INTER_AREA) # INTER_CUBIC
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.org_size_l[..., np.newaxis], ab_win), axis=2)

        # rgb2lab는 [0,1] normalize값을 넣지 않는데, lab2rgb에서 output은 [0,1] normalize값임
        pred_lab = lab2rgb(torch.from_numpy(pred_lab.copy()).type(torch.FloatTensor).unsqueeze(0).permute(0,3,1,2))
        pred_lab = pred_lab.permute(0,2,3,1).squeeze(0).numpy()
        pred_rgb = (np.clip(pred_lab, 0, 255)).astype('uint8')
        # pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        # pred_rgb = (np.clip(pred_lab, 0, 1) * 255).astype('uint8')

        self.result = pred_rgb
        
    def save_result(self, mask = None, idx=None, input_rgb=None):

        self.read_image(self.image_file)
        self.compute_result(mask = mask, idx=idx, input_rgb=input_rgb)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        return result_bgr

    def save_line_result(self, mask, req):
        """line 그려서 결과"""
        # original image
        self.read_image(self.image_file)

        # compute result
        np.set_printoptions(suppress=True)
        urllib.request.urlretrieve(req, self.image_file)
        self.compute_result(mask=mask, line_col=True, line_img_path=self.image_file)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        return result_bgr

app = flask.Flask(__name__)
# model = None
def save_log(req):
    botUserKey = req['userRequest']['user']['properties']['botUserKey']
    plusfriendUserKey = req['userRequest']['user']['properties']['plusfriendUserKey']
    time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    location_ = req['userRequest']['timezone']
    text_ = req['userRequest']['utterance']
    with open("./log/log.txt", "a") as f:
        f.write(f"{botUserKey},{plusfriendUserKey},{time_},{location_},{text_}\n")
        
def get_model(device='cpu', model_type = '채색해줘'):
    # global model_
    model = create_model(
        'icolorit_base_4ch_patch16_224',
        pretrained=False,
        drop_path_rate=0.0,
        drop_block_rate=None,
        use_rpb=True,
        avg_hint=True,
        head_mode='cnn',
        mask_cent=False,
    )
    
    model.to(device)
    checkpoint = torch.load('./pretrain/icolorit_noise.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model
    
# define col model
col_model = get_model(device)

def run_colorization(req, model, device):
    # 이미지 전처리 - 발화가 jpg, png 확장자일 때만 실행
    user_key = req['userRequest']['user']['properties']['plusfriendUserKey']
    req = req['userRequest']['utterance']
    if 'jpg' in req or 'png' in req or 'jpeg' in req or 'JPG' in req or 'JPEG' in req or 'webp' in req or 'WEBP' in req:
        np.set_printoptions(suppress=True)
        img_path = f'./sample_image/temp_user/{user_key}_img'
        urllib.request.urlretrieve(req, img_path)
        
        draw = Draw(model=model, image_file=img_path, device=device)
        colorized_img = draw.save_result()
        # 카카오 웹페이지 파일명으로 지정
        fn = '+'.join(req.split('talkm')[1:][0].split('/')[1:])
        saved_img_fn = f'./output_img/colorize_img/{fn}'
        path_, ext = os.path.splitext(saved_img_fn)
        saved_img_fn = path_ +'.png'
        
        cv2.imwrite(saved_img_fn, colorized_img)
        if not args.debug:
            gcp_link = upload_img2gcp(saved_img_fn, fn)

        msg = "채색 결과"
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "basicCard": {
                            "title": msg,
                            "description": "",
                            "thumbnail": {
                                "imageUrl": gcp_link
                            },
                            "buttons": [
                                {
                                    "action":  "webLink",
                                    "label": "사진보기",
                                    "webLinkUrl": gcp_link
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
    else:
        # simple text format
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "이미지를 보내주세요"
                        }
                    }
                ]
            }
        }
    print(res)
    return flask.jsonify(res)

# inference clip
clip_model, preprocess = clip.load("RN50", device='cuda') # ViT-B/32 , RN50

def inference_clip(model, image, text):
    with torch.no_grad():
        image_features = model.encode_image(image) # 1, 512
        text_features = model.encode_text(text) # 4, 77

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy() # 1,4
        
    return probs

def query2image(q, img_path, tile, tile_overlap, device, clip_model):
    query_word = q

    script = ['hair', 'face', 'arms', 'background', 'eye', 'legs',
             'lips', 'nose', 'ears', 'finger', 'foot', 'else','eyebrow' ,
              'petal',  'shirts', 'cloth', 'pants', 'road', 'sky', 'tree', 'top clothes', 'bottom clothes']

    text = clip.tokenize(script).to(device)
    # text = clip.tokenize([query_word]).to(device)

    tile, tile_overlap, window_size = tile, tile_overlap, 8

    img = Image.open(img_path)
    img_lq = np.array(img)
    # img_lq = preprocess(img_lq).unsqueeze(0).to(device)
    h,w,c = img_lq.shape
    tile = min(h,w)//4
    tile_overlap = int(tile//2)
    
    b=1
    tile = min(tile, h, w)
    # assert tile % window_size == 0, "tile size should be a multiple of window_size"
    tile_overlap = tile_overlap
    # sf = scale

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(h, w, 1)#.to(device)#.type_as(img_lq)
    W = torch.zeros_like(E)#.to(device)

    # batch size version
    batch_patch = []
    batch_output = []
    idx_ = []
    for n, h_idx in enumerate(h_idx_list):
        for w_idx in w_idx_list:
            in_patch = img_lq[h_idx:h_idx+tile, w_idx:w_idx+tile, :] # image to patch
            in_patch2 = Image.fromarray(in_patch) # np to pil
            in_patch2 = preprocess(in_patch2).unsqueeze(0).to(device)
            idx_.append([h_idx, w_idx])
            
            batch_patch.append(in_patch2)
            if len(batch_patch) == 128:
                in_patch2 = torch.cat(batch_patch, dim=0)
                probs = inference_clip(clip_model, in_patch2, text)
                batch_output.append(probs)
                batch_patch = []
    
    if len(batch_patch)!=0:
        in_patch2 = torch.cat(batch_patch, dim=0)
        probs = inference_clip(clip_model, in_patch2, text)
        batch_output.append(probs)
                
        
    batch_output = np.concatenate(batch_output, axis=0)[:, np.newaxis, :] # 1870, 20
    for probs, hw_idx in zip(batch_output, idx_):
        h_idx = hw_idx[0]
        w_idx = hw_idx[1]
        E[h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(probs[:,script.index(query_word)][0])
        W[h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(1)
    output = E.div_(W)
    
    return output

def run_col_transfer(req, model, clip_model, device, last_text):
    user_key = req['userRequest']['user']['properties']['plusfriendUserKey']
    req = req['userRequest']['utterance']
    try:
        img_path = f'./sample_image/temp_user/{user_key}_img'

        np.set_printoptions(suppress=True)
        urllib.request.urlretrieve(req, img_path)

        query, col_query = last_text

        output = 0
        tile_list = [64, 128, 224]
        for t in tile_list:
            to = t//2
            if t ==64:
                to = 16
            temp_output = query2image(q=query, img_path=img_path,
                            tile = t, tile_overlap=to, device=device, clip_model = clip_model)
            
            output+=temp_output/len(tile_list)


        output = F.interpolate(output.permute(2,0,1).unsqueeze(0), size=(224, 224), mode='nearest')
        output = output[0].permute(1,2,0)
        # img_lq = cv2.resize(img_lq, dsize=(224,224))

        threshold = output.quantile(0.95)
        # if query == 'hair':
        #     threshold = output.quantile(0.95)

        """1. 가장 확률 높은 부분 포지션 찾기 랜덤."""

        x = torch.where(output >threshold)[0].cpu().numpy()
        y = torch.where(output >threshold)[1].cpu().numpy()
        max_prob_pos = [(y1, x1) for x1, y1 in zip(x,y)]

        np.random.shuffle(max_prob_pos)

        """2. 픽셀 부분 찾기"""
        mask = np.zeros((224, 224, 3), np.uint8)
        mask2 = np.zeros((224, 224, 1), np.uint8)

        tl = max_prob_pos[0]
        br = (tl[0]+1, tl[1]+1)

        # print(tl, br)
        rand_color = np.random.choice(np.arange(60,128), 1)[0]
        if col_query == '빨간색':
            input_rgb = np.array([rand_color,0,0]).astype(np.uint8)
        elif col_query == '초록색':
            input_rgb = np.array([0,rand_color,0]).astype(np.uint8)
        elif col_query == '파란색':
            input_rgb = np.array([0,0,rand_color]).astype(np.uint8)
        # elif col_query == '랜덤색':
        #     c = [np.random.choice(np.arange(50,255), 1)[0], np.random.choice(np.arange(50,255), 1)[0], np.random.choice(np.arange(50,255), 1)[0]]
        #     # np.random.shuffle(c)
        #     print('color', c)
        #     input_rgb = np.array(c).astype(np.uint8)

        # lightness = 15#int((img_L[tl, br][0] + img_L[tl, br][1])/2)
        # color_ = tuple(snap_ab(lightness, input_rgb).tolist())
        # a = cv2.rectangle(mask, tl, br, color_, -1)
        # b = cv2.rectangle(mask2, tl, br, 255, -1)

        draw = Draw(model=model, image_file=img_path, device='cpu', color=True)
        col_transfer_img = draw.save_result(mask = (mask,mask2), idx=(tl, br), input_rgb=input_rgb)

        # 카카오 웹페이지 파일명으로 지정
        fn = '+'.join(req.split('talkm')[1:][0].split('/')[1:])
        saved_img_fn = f'./output_img/colorize_transfer_img/{fn}'
        path_, ext = os.path.splitext(saved_img_fn)
        saved_img_fn = path_ +'.png'
        
        cv2.imwrite(saved_img_fn, col_transfer_img)
        if not args.debug:
            gcp_link = upload_img2gcp(saved_img_fn, fn)

        msg = "채색 결과"
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "basicCard": {
                            "title": msg,
                            "description": "",
                            "thumbnail": {
                                "imageUrl": gcp_link
                            },
                            "buttons": [
                                {
                                    "action":  "webLink",
                                    "label": "사진보기",
                                    "webLinkUrl": gcp_link
                                }
                            ]
                        }
                    }
                ]
            }
        }

    except:
        # simple text format
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "채색할 이미지를 보내주세요"
                        }
                    }
                ]
            }
        }
    print(res)
    return flask.jsonify(res)

def run_col_line_transfer(req, model):
    user_key = req['userRequest']['user']['properties']['plusfriendUserKey']
    req = req['userRequest']['utterance']
    try:
        img_path = f'./sample_image/temp_user/{user_key}_img.png'

        # np.set_printoptions(suppress=True)
        # urllib.request.urlretrieve(req, img_path)

        mask = np.zeros((224, 224, 3), np.uint8)
        mask2 = np.zeros((224, 224, 1), np.uint8)

        draw = Draw(model=model, image_file=img_path, device='cpu', color=True)
        col_transfer_img = draw.save_line_result(mask = (mask,mask2), req=req)

        # 카카오 웹페이지 파일명으로 지정
        fn = '+'.join(req.split('talkm')[1:][0].split('/')[1:])
        saved_img_fn = f'./output_img/line_colorize_transfer_img/{fn}'
        path_, ext = os.path.splitext(saved_img_fn)
        saved_img_fn = path_ +'.png'
        
        cv2.imwrite(saved_img_fn, col_transfer_img)
        if not args.debug:
            gcp_link = upload_img2gcp(saved_img_fn, fn)

        msg = "채색 결과"
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "basicCard": {
                            "title": msg,
                            "description": "",
                            "thumbnail": {
                                "imageUrl": gcp_link
                            },
                            "buttons": [
                                {
                                    "action":  "webLink",
                                    "label": "사진보기",
                                    "webLinkUrl": gcp_link
                                }
                            ]
                        }
                    }
                ]
            }
        }

    except:
        # simple text format
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "채색할 이미지를 보내주세요"
                        }
                    }
                ]
            }
        }
    print(res)
    return flask.jsonify(res)
    
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route("/api/predict", methods=["POST"])
def api_predict():

    # UserRequest 중 발화를 req에 parsing.
    req = flask.request.get_json()
    save_log(req)
    user_key = req['userRequest']['user']['properties']['plusfriendUserKey']
    # req = req['userRequest']['utterance']
    # 마지막 text 보기
    df = pd.read_csv('./log/log.txt', names=['botUserKey','plusfriendUserKey','time','location', 'text'], encoding='euc-kr')

    try :
        df2 = df[df['plusfriendUserKey']==user_key]

        last_text2 = df2.iloc[-3]['text'].split('.')[-1]
        last_text = df2.iloc[-2]['text'].split('.')[-1]
        present_text = df2.iloc[-1]['text'].split('.')[-1]

        print('last_text:',last_text)
        print('last_last_text:',last_text2)

        
        if last_text=='채색해줘': # 만약 여기가 채색해줘이면?
            res = run_colorization(req, model=col_model, device='cpu')

        # elif last_text == '얼굴복원해줘':
        #     res = run_reconstruction(req, device= device_gpu)

        # elif last_text == '화질개선해줘':
        #     res = run_sr_denoise(req, task='real_sr', model=sr_model, device = device_gpu)

        # elif last_text == '노이즈제거해줘':
        #     res = run_sr_denoise(req, task='gray_dn', model=denoise_model, device= device_gpu)

        elif '색칠해줘' in last_text:
            print('run text colorization')
            obj_text, col_text, _ = last_text.split(' ')
            obj_text = {'머리':'hair', '배경':'background', '상의':'top clothes', '하의' : 'bottom clothes'}[obj_text] # , 'top clothes', 'bottom clothes'
            col_text = col_text.replace('으로', '')
            res = run_col_transfer(req, model=col_model, clip_model=clip_model, device=device_gpu, last_text = (obj_text, col_text))

        elif '선으로 이미지편집' == present_text:
            res = {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": "원본 이미지와 편집된 이미지를 차례로 올려주세요. (현재는 한가지 색상만 지원됩니다.)"
                            }
                        }
                    ]
                }
            }
        elif '선으로 이미지편집' == last_text2:
            print('run line colorization')
            res = run_col_line_transfer(req, col_model)

        else :
            try:
                # user_key = req['userRequest']['user']['properties']['plusfriendUserKey']
                req = req['userRequest']['utterance']
                img_path = f'./sample_image/temp_user/{user_key}_img.png'

                np.set_printoptions(suppress=True)
                urllib.request.urlretrieve(req, img_path)
                print('이미지 미리 저장')

                return {}

            except:
                res = {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {
                                "simpleText": {
                                    "text": "사용하실 서비스를 메뉴에서 선택해주세요."
                                }
                            }
                        ]
                    }
                }
    except: # 처음 들어온 유저
        return {}
    
    return res

        
if __name__ == "__main__":
    # res = run_col_transfer2(model=col_model, clip_model=clip_model, device=device_gpu, last_text = ('hair', '빨간색'))
    print("* Loading Pytorch model and Flask starting server...")
    print("please wait until server has fully started")
    
    app.run(host='0.0.0.0')
