import argparse
import os
import os.path as osp
import random
import glob
from tqdm import tqdm


def make_fixed_hint():
    parser = argparse.ArgumentParser(description="Making fixed hint set for interactive colorization")
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--hint_dir', type=str)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--hint_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    assert args.img_size % args.hint_size == 0
    filenames = sorted(os.listdir(args.img_dir))
    if 'imagenet/val' in args.img_dir:
        filenames = sorted(glob.glob(os.path.join(args.img_dir, '*/*')))
        filenames = [x.split('/')[-1] for x in filenames if x.split('/')[-1].split('.')[-1] in ['png', 'jpg', 'jpeg', 'JPEG']]
    random.seed(args.seed)
    # '1', '2', '5', '10', '20', '50', '100', '200', '500'
    for num_hint in [10]:#[1, 2, 5, 10, 20, 50]:#[0]:#[1,5,10,20,50,100]:#[10]:
        for file in tqdm(filenames):
            lines = [f'{random.randint(0, args.img_size//args.hint_size - 1) * args.hint_size} '
                     f'{random.randint(0, args.img_size//args.hint_size - 1) * args.hint_size}\n'
                     for _ in range(num_hint)]
            txt_file = osp.join(args.hint_dir, str(args.seed), f'h{args.hint_size}-n{num_hint}',
                                osp.splitext(file)[0] + '.txt')
            os.makedirs(osp.dirname(txt_file), exist_ok=True)
            with open(txt_file, 'w') as f:
                f.writelines(lines)


if __name__ == '__main__':
    make_fixed_hint()

# python preparation/make_mask.py --img_dir /home/data/imagenet/ctest10k/ --hint_dir ./data/ctest10k
# python preparation/make_mask.py --img_dir /home/data/oxford102_flowers/ --hint_dir ./data/oxford102_flowers
# python preparation/make_mask.py --img_dir /home/data/cub_200 --hint_dir ./data/cub_200

# sample (for paper)
# python preparation/make_mask.py --img_dir ./data/gt --hint_dir ./data/gt_txt
# python preparation/make_mask.py --img_dir ./data/518 --hint_dir ./data/518_txt
# python preparation/make_mask.py --img_dir ./data/518_2 --hint_dir ./data/518_2_txt

# sample (for ijcai paper)
# python preparation/make_mask.py --img_dir ./data/ijcai_quality/ --hint_dir ./data/ijcai_quality_txt

# color (deepfasion)
# python preparation/make_mask.py --img_dir --hint_dir ./data/deepfasion_txt
# python preparation/make_mask.py --img_dir /home/data/images1024x1024 --hint_dir ./data/images1024x1024_txt



