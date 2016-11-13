'''
Requeriments:
- Python (We use 3.5).
- Packages: scipy, matplotlib, cv2, skimage, numpy.
'''

from scipy.ndimage import imread
from matplotlib.pyplot import imshow, show, figure, subplot, suptitle
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage import transform
from skimage.morphology import closing, opening, dilation, disk
from skimage.measure import label, find_contours, grid_points_in_poly
import numpy as np
from skimage.draw import ellipse
from scipy.io import loadmat
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import active_contour


def rgb2gray(rgb):
    gray = 0.2989 * rgb[..., 0]
    gray[:] += 0.5870 * rgb[..., 1]
    gray[:] += 0.1140 * rgb[..., 2]
    return gray

# Step 1
I = imread('imgs/IMD426.bmp')

# Step 2
IGray = rgb2gray(I)
IGaussian = cv.GaussianBlur(IGray, (9, 9), 0.5)

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

#skinMoleClosing0 = closing(skinMoleC, selem=data['se']['se0'][0][0])
skinMoleClosing4 = closing(skinMoleC, selem=data['se']['se4'][0][0])
#skinMoleClosing6 = closing(skinMoleC, selem=data['se']['se6'][0][0])

# Step 6

#skinMoleOpening0 = opening(skinMoleClosing0, selem=data['se']['se0'][0][0])
skinMoleOpening4 = opening(skinMoleClosing4, selem=data['se']['se4'][0][0])
#skinMoleOpening6 = opening(skinMoleClosing6, selem=data['se']['se6'][0][0])

# Step 7

#skinMoleFillHoles0 = binary_fill_holes(skinMoleOpening0, structure=data['se']['se0'][0][0])
skinMoleFillHoles4 = binary_fill_holes(skinMoleOpening4, structure=data['se']['se4'][0][0])
#skinMoleFillHoles6 = binary_fill_holes(skinMoleOpening6, structure=data['se']['se6'][0][0])

#Step 8

'''
contours0 = find_contours(skinMoleFillHoles0, 0.5)[0]
contours0 = contours0[0:-1, :]
skinMoleSnakes0 = active_contour(skinMoleC, contours0, bc='free')
skinMoleSnakes0 = grid_points_in_poly(I.shape, skinMoleSnakes0)
'''

contours4 = find_contours(skinMoleFillHoles4, 0.5)[0]
contours4 = contours4[0:-1, :]
skinMoleSnakes4 = active_contour(skinMoleC, contours4, bc='free')
skinMoleSnakes4 = grid_points_in_poly(I.shape, skinMoleSnakes4)

'''
contours6 = find_contours(skinMoleFillHoles6, 0.5)[0]
contours6 = contours6[0:-1, :]
skinMoleSnakes6 = active_contour(skinMoleC, contours6, bc='free')
skinMoleSnakes6 = grid_points_in_poly(I.shape, skinMoleSnakes6)
'''

# Step 9

'''
ILabel0 = label(skinMoleSnakes0)
skinMoleOpenArea0 = np.zeros((I.shape[0:2]), dtype=np.uint8)

if ILabel0.max() > 1:
    for i in range(1, ILabel0.max()):
        if np.sum(ILabel0 == i) > 50:
            skinMoleOpenArea0 += ILabel0 == i
else:
    skinMoleOpenArea0 = skinMoleSnakes0
'''

ILabel4 = label(skinMoleSnakes4)
skinMoleOpenArea4 = np.zeros((I.shape[0:2]), dtype=np.uint8)

if ILabel4.max() > 1:
    for i in range(1, ILabel4.max()):
        if np.sum(ILabel4 == i) > 50:
            skinMoleOpenArea4 += ILabel4 == i
else:
    skinMoleOpenArea4 = skinMoleSnakes4

'''
ILabel6 = label(skinMoleSnakes6)
skinMoleOpenArea6 = np.zeros((I.shape[0:2]), dtype=np.uint8)

if ILabel6.max() > 1:
    for i in range(1, ILabel6.max()):
        if np.sum(ILabel6 == i) > 50:
            skinMoleOpenArea6 += ILabel6 == i
else:
    skinMoleOpenArea6 = skinMoleSnakes6
'''

# Step 10

#skinMoleClosing0Last = closing(skinMoleOpenArea0, selem=data['se']['se0'][0][0])
skinMoleClosing4Last = closing(skinMoleOpenArea4, selem=data['se']['se4'][0][0])
#skinMoleClosing6Last = closing(skinMoleOpenArea6, selem=data['se']['se6'][0][0])

#skinMoleOpening0Last = opening(skinMoleClosing0Last, selem=data['se']['se0'][0][0])
skinMoleOpening4Last = opening(skinMoleClosing4Last, selem=data['se']['se4'][0][0])
#skinMoleOpening6Last = opening(skinMoleClosing6Last, selem=data['se']['se6'][0][0])

mask2 = np.zeros((I.shape[0:2]), dtype=np.uint8)
rr, cc = ellipse(round(I.shape[0]/2), round(I.shape[1]/2), round(I.shape[0]/2)-1, round(I.shape[1]/2)-1)

mask2[rr, cc] = 1
mask2 = dilation(mask2, selem=disk(30))

#skinMoleSegmented0 = skinMoleOpening0Last * mask2
skinMoleSegmented4 = skinMoleOpening4Last * mask2
#skinMoleSegmented6 = skinMoleOpening6Last * mask2

# Ground truth

IS = imread('imgs/IMD426_lesion.bmp')

figure(0)
suptitle('Our - Diff - Ground truth')

subplot(1, 3, 1)
imshow(skinMoleSegmented4, cmap='gray')

subplot(1, 3, 2)
imshow(255 - ((255 - (IS * skinMoleSegmented4)) + (IS == skinMoleSegmented4)), cmap='gray')

subplot(1, 3, 3)
imshow(IS, cmap='gray')

show()
