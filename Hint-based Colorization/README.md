### Installation

```
conda create -n colorization python=3.9 -y
conda activate colorization
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Preprocess
```
python preparation/make_mask.py --img_dir /home/data/colorization/skt_test/ --hint_dir ./data/skt_test/hint
```
## For Swin
```
cd kernels/window_process
python setup.py install #--user
```

## Training

First prepare an official ImageNet dataset with the following structure. 

```
train
 └ id1
   └ image1.JPEG
   └ image2.JPEG
   └ ...
 └ id2
   └ image1.JPEG
   └ image2.JPEG
   └ ...     

```

Please fill in the train/evaluation directories in the scripts/train.sh file and execute

```
bash scripts/train.sh
```


## Inference

```
python3 icolorit_ui.py --model icolorit_tiny_4ch_patch16_224 --model_path ../saved_models/icolorit_tiny_4ch_patch16_224/icolorit_tiny_4ch_patch16_224_230122_140519/checkpoint-11.pth --target_image ./img/**.jpg
```
