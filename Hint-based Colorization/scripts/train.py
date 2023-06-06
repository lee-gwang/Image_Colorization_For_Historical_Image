import os

epochs = 25
n_gpu = 8
gpu = '0,1,2,3,4,5,6,7'
lr = 5e-5
batch_size = 128
port = 5675
model = 'icolorit_tiny_4ch_patch16_224'
os.system(f'CUDA_VISIBLE_DEVICES={gpu} python -m torch.distributed.launch --nproc_per_node={n_gpu} --master_port {port} \
    train.py \
    --data_path /home/data/imagenet/train \
    --val_data_path /home/data/imagenet/ctest10k \
    --val_hint_dir ./data/ctest10k/1234 \
    --output_dir ./saved_models/color_noise \
    --exp_name {model} \
    --save_args_txt \
    --epochs {epochs} \
    --batch_size {batch_size} \
    --model {model} \
    --lr {lr}')
    # --model_path ./pretrained/{model}.pth \