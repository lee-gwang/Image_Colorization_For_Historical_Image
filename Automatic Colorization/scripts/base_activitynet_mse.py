import os

os.system('python train.py --dataset activitynet_384 --save_img -gpu 4,5,6,7 --backbone efficientnet-b4 --scheduler CosineAnnealingLR --epoch 20 -bs 128 --loss mse -expc mse2')
