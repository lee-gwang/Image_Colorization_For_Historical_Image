import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
import os
import glob

# path = glob.glob('./output_dirs/Unet-efficientnet-b1-trash/pred/*')
# path = glob.glob('./output_dirs/ict_old/*')
path = glob.glob('/home/data/colorization/gif/gif_frames/output_img/*')


# path = [os.path.join('/'.join(x.split('/')[:-1]), f'{n}.jpg') for n,x in enumerate(path)]
path = sorted(path)
print(path)
paths = [ Image.open(i) for i in path]
imageio.mimsave('./test.gif', paths, fps=10)