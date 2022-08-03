# Experiments

## Todo
- validation 너무 느린데, 더해서 풀자
- train중에 18데이터 validation으로 함께 visualize 시켜보자
- inference 사진많으면 가끔 왜안되니..?


<!-- ### Datasets
- 6,996 video clips
- 170,362 frames
- Train 80% / Val 20%
- GroupKFold

## ActivityNet Data Path 

```
​```
${ActivityNet}}
├── 384(resized_size)
|   └── XXXXX_frames
|       └── xxxx.jpg
|       └── ...
​```
``` -->

## Preprocess Script

```bash
# 0720
## gettyimages
$ python preprocess.py --data gettyimages --data_path /home/data/colorization/gettyimages/ --size 768
python preprocess.py --data human --data_path /home/data/colorization/human --size 768 --fold 10 --depth 2
python preprocess.py --data ffhd --data_path /home/data/colorization/ffhd --size 768 --fold 10 --depth 2

python preprocess.py --data_path ./data/ffhd/ --size 1024 --fold 10 --depth 2
# no resize
python preprocess.py --data_path /home/data/colorization/street/ --fold 10 --depth 2
#
```

## Train Script
```bash

python train.py --dataset_path ./data/ffhq_v2/HR_768/train_10.csv --save_img -gpu 0,1,2,3 --backbone efficientnet-b1 --scheduler CosineAnnealingLR --epoch 50 -bs 128 -expc 0802_ffhq_v2_pretrained_noise
python train.py --dataset_path ./data/ffhq_v2/HR_768/train_10.csv --save_img -gpu 0,1,2,3 --backbone efficientnet-b1 --scheduler CosineAnnealingLR --epoch 50 -bs 64 -expc 0802_ffhq_v2_pretrained_noise --pretrained ./saved_models/Unet-efficientnet-b1-0731_ffhd_else/fold0_best.pth 

python train.py --dataset_path ./data/ffhq_v3/HR_768/train_10.csv --save_img -gpu 4,5,6,7 --backbone efficientnet-b1 --scheduler CosineAnnealingLR --epoch 50 -bs 64 -expc 0803_ffhq_v3_pretrained_noise2 --pretrained ./saved_models/Unet-efficientnet-b1-0802_ffhq_v2_pretrained_noise/fold0_best_e50.pth --val_iter 1



```
## Validation
```bash
python val.py --backbone <your_models> -sm <your_saved_models_path> --save_img -expc sample 
```

## Inference & Demo
```bash
python inference.py --backbone efficientnet-b0 -gpu 2,3 --img_path ./data/animal/ --depth=2 -sm ./saved_models/Unet-efficientnet-b0-large_dataset_384-new_384/fold0_best.pth -expc
python inference.py --backbone efficientnet-b1 -gpu 0,1,2,3 --img_path ./data/color_SR_After/ -bs 128 --depth=1 -sm ./saved_models/Unet-efficientnet-b1-gettyimages_768-0719/fold0_best.pth

# 0803
python inference_patch.py --backbone efficientnet-b1 --img_path /home/data/colorization/skt_sample/ --depth=1 -sm ./saved_models/Unet-efficientnet-b1-0803_ffhq_v3_pretrained_noise2/fold0_best_e9.pth -expc temp_power_ensemble --gpu=0,1 --image --patch

```

## Visualize (Unet-efficientnetb0)
<!-- ![ex_screenshot](./imgs/5.jpg){: width="256" height="256"}
![ex_screenshot](./imgs/14.jpg){: width="256" height="256"}
![ex_screenshot](./imgs/19.jpg){: width="256" height="256"}
![ex_screenshot](./imgs/38.jpg){: width="256" height="256"} -->

<!-- <img src="./imgs/5.jpg" width="200" height="200"> <img src="./imgs/14.jpg" width="200" height="200">

<img src="./imgs/19.jpg" width="200" height="200"> <img src="./imgs/38.jpg" width="200" height="200"> -->

<img src="./imgs/visual_activity.png" >





<!-- ## 0618-0620
1. 4x model (O)
2. activitynet 학습 (O)
3. 2x,4x,8x등 데이터셋 따로저장 만들기 (현재는 torch resize사용) (not patch version)
4. patch version으로 2x,4x,8x 만들기 / validation은 patch merging고려
5. inference 할때 이미지 들어오면 알아서 patch단위로 나누고 예측한후 merging까지 하는 코드짜기
5. 코드에서 labels to x4 이런단어로 바꿀때 단어 바꾸니까, path명중에 겹치는게 바뀌어버리네..


1. 4X 모델에 넣고, 맨 마지막 block 빼기
2. classification or regression(LAB)
3. psnr은 rgb로 고쳐놓고 재기 (현재 naive psnr임)
4. best마다만 저장되게하자 이미지
5. validatoin inference?
6. import 필요없는 라이브러리 지우기