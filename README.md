# Experiments

## Todo
- activation, loss
- normalization
- inference와 demo 간단하게 테스트 가능하게 코드 다시짜기


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
# ActivityNet dataset
# $ python frame_extract.py --dir <raw_video_path> --output <frame_image_save_path> --sampling 5
# $ python preprocess.py

# 0720
$ python preprocess.py --data gettyimages --data_path /home/data/colorization/gettyimages/ --size 768
$ python preprocess.py --data imagenet --data_path /home/data/imagenet/train --size 512
```

## Train Script
```bash
# gettyimages
$ python train.py --dataset ./data/gettyimages/HR_768/train.csv --save_img -gpu 0,1,2,3 --backbone efficientnet-b1 --scheduler CosineAnnealingLR --epoch 20 -bs 64 -expc 0719

```
<!-- python train.py --dataset gettyimages_768 --save_img -gpu 0,1,2,3 --backbone efficientnet-b0 --scheduler CosineAnnealingLR --epoch 15 -bs 128 -expc getty_768 -->
## Validatoin
```bash
python val.py --backbone <your_models> -sm <your_saved_models_path> --save_img -expc sample 
```

## Inference & Demo
```bash
python inference.py --backbone efficientnet-b0 -gpu 2,3 --img_path ./data/animal/ --depth=2 -sm ./saved_models/Unet-efficientnet-b0-large_dataset_384-new_384/fold0_best.pth 
# python inference.py --backbone efficientnet-b0 -gpu 2,3 --img_path ./data/else/ --depth=1 -sm ./saved_models/imagenet/Unet-efficientnet-b0-imagenet_sample20k-sample30-inputL/fold0_best.pth -expc else

# python inference.py --backbone efficientnet-b0 -gpu 2,3 --img_path /home/data/colorization/gettyimages/ --depth=2 -sm ./saved_models/imagenet/Unet-efficientnet-b0-imagenet_sample20k-sample30-inputL/fold0_best.pth -expc getty
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