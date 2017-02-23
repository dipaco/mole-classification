from matplotlib.pyplot import imshow, subplot, title, suptitle, show, figure, imread, colorbar
import os
import fnmatch
from skimage.color import rgb2lab, rgb2gray, gray2rgb
from skimage.util import dtype
import numpy as np
from PH2Dataset import PH2Dataset
from skimage.filters import threshold_otsu
from skimage.draw import ellipse
import cv2

def get_mask(s):
    mask = np.zeros(s, dtype=np.uint8)
    rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 2) - 1, round(s[1] / 2) - 1)
    mask[rr, cc] = 1
    return mask


dataset = PH2Dataset('PH2Dataset')
for image_idx in range(dataset.num_images):

    image = dataset.image_names[image_idx]

    IOriginal = dataset.get_image_data(image_idx)

    # Gets the mask to avoid dark areas in segmentation
    mask = get_mask(IOriginal.shape[0:2])
    I = gray2rgb(mask) * IOriginal
    GT = (rgb2gray(dataset.get_ground_truth_data(image_idx).astype(float)) * mask) > 120
    IGray = rgb2gray(I)
    thresh = threshold_otsu(IGray)
    IOtsu = IGray <= thresh
    IOtsu = np.logical_and(IOtsu, mask)

    Ilab = rgb2lab(I)


    '''
    subplot(2, 3, 1)
    title('R')
    colorbar(imshow(IOriginal[:, :, 0]))
    subplot(2, 3, 2)
    title('G')
    colorbar(imshow(IOriginal[:, :, 1]))
    subplot(2, 3, 3)
    title('B')
    colorbar(imshow(IOriginal[:, :, 2]))
    subplot(2, 3, 4)
    title('L')
    colorbar(imshow(Ilab[:, :, 0]))
    subplot(2, 3, 5)
    title('A')
    colorbar(imshow(Ilab[:, :, 1]))
    subplot(2, 3, 6)
    title('B')
    colorbar(imshow(Ilab[:, :, 2]))
    '''

    print('{:15} {:15} {:15} {:15}'.format(' ', 'R', 'G', 'B'))
    subplot(1, 2, 1)
    ii, jj = np.where(GT)
    Ii = I[ii, jj, 0].ravel()

    ii, jj = np.where(np.logical_and(np.logical_not(GT), mask))
    Io = I[ii, jj, 0].ravel()

    colorbar(imshow(I[:, :, 0] * GT))
    miniR = Ii.min()
    maxiR = Ii.max()

    subplot(1, 2, 2)
    colorbar(imshow(I[:, :, 0] * np.logical_not(GT)))
    minoR = Io.min()
    maxoR = Io.max()

    #show()

    subplot(1, 2, 1)


    colorbar(imshow(I[:, :, 1] * GT))
    miniG = Ii.min()
    maxiG = Ii.max()

    subplot(1, 2, 2)
    colorbar(imshow(I[:, :, 1] * np.logical_not(GT)))
    minoG = Io.min()
    maxoG = Io.max()

    #show()

    subplot(1, 2, 1)

    colorbar(imshow(I[:, :, 2] * GT))
    miniB = Ii.min()
    maxiB = Ii.max()

    subplot(1, 2, 2)
    colorbar(imshow(I[:, :, 2] * np.logical_not(GT)))
    minoB = Io.min()
    maxoB = Io.max()

    #show()


    print('{:15} {:15} {:15} {:15}'.format('Min i', miniR, miniG, miniB))
    print('{:15} {:15} {:15} {:15}'.format('Max i', maxiR, maxiG, maxiB))
    print('{:15} {:15} {:15} {:15}'.format('Min o', minoR, minoG, minoB))
    print('{:15} {:15} {:15} {:15}'.format('Max o', maxoR, maxoG, maxoB))



    print('{:15} {:15} {:15} {:15}'.format(' ', 'L', 'A', 'B'))
    subplot(1, 2, 1)
    ii, jj = np.where(GT)
    Ii = Ilab[ii, jj, 0].ravel()

    ii, jj = np.where(np.logical_and(np.logical_not(GT), mask))
    Io = Ilab[ii, jj, 0].ravel()

    colorbar(imshow(Ilab[:, :, 0] * GT))
    miniR = Ii.min()
    maxiR = Ii.max()

    subplot(1, 2, 2)
    colorbar(imshow(Ilab[:, :, 0] * np.logical_not(GT)))
    minoR = Io.min()
    maxoR = Io.max()

    # show()

    subplot(1, 2, 1)

    colorbar(imshow(Ilab[:, :, 1] * GT))
    miniG = Ii.min()
    maxiG = Ii.max()

    subplot(1, 2, 2)
    colorbar(imshow(Ilab[:, :, 1] * np.logical_not(GT)))
    minoG = Io.min()
    maxoG = Io.max()

    # show()

    subplot(1, 2, 1)

    colorbar(imshow(Ilab[:, :, 2] * GT))
    miniB = Ii.min()
    maxiB = Ii.max()

    subplot(1, 2, 2)
    colorbar(imshow(Ilab[:, :, 2] * np.logical_not(GT)))
    minoB = Io.min()
    maxoB = Io.max()

    # show()


    print('{:15} {:15} {:15} {:15}'.format('Min i', miniR, miniG, miniB))
    print('{:15} {:15} {:15} {:15}'.format('Max i', maxiR, maxiG, maxiB))
    print('{:15} {:15} {:15} {:15}'.format('Min o', minoR, minoG, minoB))
    print('{:15} {:15} {:15} {:15}'.format('Max o', maxoR, maxoG, maxoB))

    # show()
