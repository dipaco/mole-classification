'''
Requeriments:
- Python (We use 3.5).
- Packages: scipy, matplotlib, cv2, skimage, numpy.
'''

from scipy.ndimage import imread
from matplotlib.pyplot import imshow, show, title, figure, subplot, suptitle
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage import transform
from skimage.morphology import erosion, dilation, disk, square, diamond, closing, black_tophat, white_tophat, opening
from skimage.measure import label
import numpy as np
from skimage.draw import ellipse
from scipy.io import loadmat
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import compare_nrmse
from skimage import img_as_ubyte

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
mask1 = np.zeros((I.shape[0:2]), dtype=np.uint8)
rr, cc = ellipse(round(I.shape[0]/2), round(I.shape[1]/2), round((I.shape[0]/2)*(11/12.0)), round((I.shape[1]/2)*(11/12.0)))
mask1[rr, cc] = 1

skinMoleC = mask1 * IOtsu
# imshow(skinMoleC, cmap="gray")

# Step 5
'''
# Temporal disk, it was not successful
disk11 = disk(11)
skinMoleClosing = closing(skinMoleC, selem=disk11)
'''
# Disks created by radial decomposition using periodic lines
data = loadmat('se.mat')

skinMoleClosing0 = closing(skinMoleC, selem=data['se']['se0'][0][0])
skinMoleClosing4 = closing(skinMoleC, selem=data['se']['se4'][0][0])
skinMoleClosing6 = closing(skinMoleC, selem=data['se']['se6'][0][0])

skinMoleOpening0 = opening(skinMoleClosing0, selem=data['se']['se0'][0][0])
skinMoleOpening4 = opening(skinMoleClosing4, selem=data['se']['se4'][0][0])
skinMoleOpening6 = opening(skinMoleClosing6, selem=data['se']['se6'][0][0])

# Ground truth
IS = imread('imgs/IMD242_lesion.bmp')

figure(0)
suptitle('Our - Diff - Ground truth')

subplot(1, 3, 1)
imshow(skinMoleOpening0, cmap='gray')

subplot(1, 3, 2)
imshow(255 - ((255 - (IS * skinMoleOpening0)) + (IS == skinMoleOpening0)), cmap='gray')

subplot(1, 3, 3)
imshow(IS, cmap='gray')

show()
