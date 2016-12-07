import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
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
        rr, cc = ellipse((s[0] / 2), (s[1] / 2), (s[0] / 2) -1 , (s[1] / 2) -1 )
        mask[rr, cc] = 1
        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(IOriginal)
        plt.show()
        return mask

    mask = get_mask(IOriginal.shape[0:2])
