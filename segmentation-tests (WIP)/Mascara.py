import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from skimage.draw import ellipse
import fnmatch
import os

path = 'imgs'

for image in fnmatch.filter(os.listdir('imgs'), '*.bmp'):

    IOriginal = imread(path + '/' + image)
    def get_mask(s):
        plt.figure()
        mask = np.zeros(s, dtype=np.uint8)
        rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 2) , round(s[1] / 2) )
        mask[rr, cc] = 1
        plt.imshow(mask)
        plt.show()
        return mask

    mask = get_mask(IOriginal.shape[0:2])