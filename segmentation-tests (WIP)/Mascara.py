import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from skimage.color import rgb2gray
from skimage.draw import ellipse
import fnmatch
import os

#testing purpose only

path = 'imgs'

for image in fnmatch.filter(os.listdir('imgs'), '*.bmp'):

    IOriginal = imread(path + '/' + image)
    def get_mask(s):
        plt.figure()
        mask = np.zeros(s, dtype=np.uint8)
        rr, cc = ellipse((s[0] / 2), (s[1] / 2), (s[1] / 2) -1, (s[1] / 2) -1)
        ii = np.where(
            np.logical_and(np.logical_and(np.logical_and(rr >= 0, rr < s[0]), cc >= 0), cc < s[1])
        )[0]
        mask[rr[ii], cc[ii]] = 1
        plt.subplot(221)
        plt.imshow(mask)
        plt.subplot(222)
        plt.imshow(IOriginal)
        plt.subplot(223)
        plt.imshow(rgb2gray(IOriginal.astype(float)) * mask, cmap='gray')
        plt.show()
        return mask

    mask = get_mask(IOriginal.shape[0:2])
