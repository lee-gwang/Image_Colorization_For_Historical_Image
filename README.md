# ImageNet Experiments
## ImageNet Data Path 

```
​```
${ImageNet}}
├── train
|   └── XXX.JPG
|   └── ...
├── test
|   └── XXX.JPG
|   └── ...

​```
```

## Preprocess Script

```bash
# ImageNet 200K sampling
$ python preprocess.py
```

## Train Script
```bash
# baseline
# 추후에 더 추가 예정
$ python ./script/base_imgenet20k.py
```

## Inference & Demo
```bash
# 추후에 추가 예정
# 현재는 train에서 --show_img를 통해 확인
```

## Visualize (Unet-efficientnetb0)
<!-- ![ex_screenshot](./imgs/5.jpg){: width="256" height="256"}
![ex_screenshot](./imgs/14.jpg){: width="256" height="256"}
![ex_screenshot](./imgs/19.jpg){: width="256" height="256"}
![ex_screenshot](./imgs/38.jpg){: width="256" height="256"} -->

<img src="./imgs/5.jpg" width="300" height="300">
<img src="./imgs/14.jpg" width="300" height="300">

<img src="./imgs/19.jpg" width="300" height="300">
<img src="./imgs/38.jpg" width="300" height="300">





# TODO
1. 4X 모델에 넣고, 맨 마지막 block 빼기
2. classification or regression(LAB)
3. psnr은 rgb로 고쳐놓고 재기 (현재 naive psnr임)
4. best마다만 저장되게하자 이미지
5. validatoin inference?
6. import 필요없는 라이브러리 지우기