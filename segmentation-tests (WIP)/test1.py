'''
Requeriments:
- Python (We use 3.5).
- Packages: scipy, matplotlib, cv2, skimage, numpy.
'''
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage import transform
from skimage.morphology import erosion, dilation, disk, square, diamond, closing
from skimage.measure import label
import numpy as np
from skimage.draw import ellipse

def rgb2gray(rgb):
    gray = 0.2989 * rgb[..., 0]
    gray[:] += 0.5870 * rgb[..., 1]
    gray[:] += 0.1140 * rgb[..., 2]
    return gray

# Step 1
I = imread('imgs/IMD242.bmp')

# Step 2
IGray = rgb2gray(I)
IGaussian = cv.GaussianBlur(IGray,(9,9), 0.5)

# Step 3
thres = threshold_otsu(IGaussian)
IOtsu = IGaussian <= thres

# Step 4
'''
# mn is all images mean
mn = transform.resize(rgb2gray(imread('mn.png')), (I.shape[0], I.shape[1]))
thres = threshold_otsu(mn)
mask = mn <= thres

maskLabel = label(mask)
mx = 0
i1 = 0

for i in range(1, maskLabel.max()):
    tmp = np.sum(maskLabel == i)
    if mx < tmp:
        mx = tmp
        i1 = i

# Method 1: center removal
maskCorners1 = dilation(erosion(mask - (maskLabel == i1), selem=disk(5)), selem=disk(40))
#                                                                    ^               ^^

# Method 2: center expansion
maskCorners2 = dilation((maskLabel == i1), selem=disk(10))
#                                                     ^^

skinMoleA = maskCorners >= IOtsu
skinMoleB = maskCorners < IOtsu
'''

# Method 3: ellipse
mask1 = np.zeros((I.shape[0], I.shape[1]), dtype=np.uint8)
rr, cc = ellipse(round(I.shape[0]/2), round(I.shape[1]/2), round((I.shape[0]/2)*(11/12)), round((I.shape[1]/2)*(11/12)))
mask1[rr, cc] = 1

skinMoleC = mask1 * IOtsu
# plt.imshow(skinMoleC, cmap="gray")

'''
# Step 5
# Temporal disk, it was not successful
disk11 = disk(11)
skinMoleClosing = closing(skinMoleC, selem=disk11)
'''

plt.show()
