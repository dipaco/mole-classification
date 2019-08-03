# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import rescale
from skimage.color import rgb2gray, label2rgb, gray2rgb, rgb2lab, rgb2hsv
from balu.DataSelectionAndGeneration import Bds_nostratify
from balu.FeatureExtraction import Bfx_haralick, Bfx_geo, Bfx_basicgeo, Bfx_lbp
from skimage.exposure import histogram
from scipy.special import entr
from skimage.feature import multiblock_lbp
from skimage.measure import label, regionprops
from matplotlib.pyplot import imshow, show, colorbar, bar


def get_features(I, Isegmented, method='pennisi'):

    if method == 'pennisi':
        return get_features_pennisi(I, Isegmented)
    else:
        return getFeatures_old(I, Isegmented)


def get_features_pennisi(I, Isegmented):
    Xstack = np.zeros((1, 3*16 + 3))
    Xnstack = []
    HSV = rgb2hsv(I) * np.stack(3*(Isegmented,), 2)
    #imshow(HSV[:, :, 2])
    #colorbar()
    #show()
    h, _ = np.histogram(HSV[:, :, 0].ravel(), 16, range=(0, 1))
    s, _ = np.histogram(HSV[:, :, 1].ravel(), 16, range=(0, 1))
    v, _ = np.histogram(HSV[:, :, 2].ravel(), 16, range=(0, 1))
    Xstack[0, 0:16] = h
    Xstack[0, 16:32] = s
    Xstack[0, 32:48] = v
    #bar(np.arange(16), v)
    #show()
    Xnstack.extend(['H bin {}'.format(i) for i in range(16)])
    Xnstack.extend(['S bin {}'.format(i) for i in range(16)])
    Xnstack.extend(['V bin {}'.format(i) for i in range(16)])

    options = {'b': [
        {'name': 'basicgeo', 'options': {'show': False}},  # basic geometric features
        #{'name': 'hugeo', 'options': {'show': False}},  # Hu moments
        #{'name': 'flusser', 'options': {'show': False}},  # Flusser moments
        #{'name': 'fourierdes', 'options': {'show': False, 'Nfourierdes': 12}},  # Fourier descriptors
    ]}

    Xtmp, Xntmp = Bfx_geo(Isegmented, options)

    Xstack[0, 48:] = Xtmp[0, [13, 16, 17]]
    Xnstack.extend(['Solidity', 'Convex Area', 'Filled Area'])
    return Xstack, np.array(Xnstack)


def getFeatures_old(I, Isegmented):

    Ilab = rgb2lab(I)
    Ilab[:, :, 0] *= 2.55
    Ilab[:, :, 1] += 127
    Ilab[:, :, 2] += 128

    I05 = rescale(I, 0.5)
    I025 = rescale(I, 0.25)
    I0125 = rescale(I, 0.125)
    Ilab05 = rescale(Ilab, 0.5)
    Ilab025 = rescale(Ilab, 0.25)
    Ilab0125 = rescale(Ilab, 0.125)
    Isegmented05 = rescale(Isegmented, 0.5)
    Isegmented025 = rescale(Isegmented, 0.25)
    Isegmented0125 = rescale(Isegmented, 0.125)
    Xstack = []
    Xnstack = []

    options = {'b': [
        {'name': 'basicgeo', 'options': {'show': False}},  # basic geometric features
        {'name': 'hugeo', 'options': {'show': False}},  # Hu moments
        {'name': 'flusser', 'options': {'show': False}},  # Flusser moments
        {'name': 'fourierdes', 'options': {'show': False, 'Nfourierdes': 12}},  # Fourier descriptors
    ]}

    Xtmp, Xntmp = Bfx_geo(Isegmented, options)
    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    options = {'dharalick': [3, 6, 12]}  # # pixels distance for coocurrence
    # options = {'dharalick': 3}  # 3 pixels distance for coocurrence
    J = rgb2gray(I.astype(float))
    Xtmp, Xntmp = Bfx_haralick(J, Isegmented, options)  # Haralick features
    Xntmp = [name + '_gray' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_haralick(Ilab[:, :, 0], Isegmented, options)  # Haralick features
    Xntmp = [name + '_L*' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_haralick(Ilab[:, :, 1], Isegmented, options)  # Haralick features
    Xntmp = [name + '_A*' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_haralick(Ilab[:, :, 2], Isegmented, options)  # Haralick features
    Xntmp = [name + '_B*' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    a, _ = histogram(np.round(rgb2gray(I.astype(float))))
    a = a / a.sum()

    Xtmp = [entr(a).sum(axis=0)]
    Xntmp = ['Entropy']

    Xstack.extend(Xtmp)
    Xnstack.extend(Xntmp)

    count = float(I[:, :, 0].size)

    mean_red = np.sum(I[:, :, 0])
    mean_green = np.sum(I[:, :, 1])
    mean_blue = np.sum(I[:, :, 2])

    Xtmp = [mean_red / count, mean_green / count, mean_blue / count]
    Xntmp = ['mean_red', 'mean_green', 'mean_blue']

    Xstack.extend(Xtmp)
    Xnstack.extend(Xntmp)

    Xstack.extend(Xtmp)
    Xnstack.extend(Xntmp)

    mean_l = np.sum(Ilab[:, :, 0])
    mean_a = np.sum(Ilab[:, :, 1])
    mean_b = np.sum(Ilab[:, :, 2])

    Xtmp = [mean_l / count, mean_a / count, mean_b / count]
    Xntmp = ['mean_l', 'mean_a', 'mean_b']

    Xstack.extend(Xtmp)
    Xnstack.extend(Xntmp)

    options = {
        'weight': 0,
        'vdiv': 3,
        'hdiv': 3,
        'samples': 8,
        'mappingtype': 'nri_uniform'
    }

    Xtmp, Xntmp = Bfx_lbp(I[:, :, 0], Isegmented, options)
    Xntmp = [name + '_red_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_lbp(I[:, :, 1], Isegmented, options)
    Xntmp = [name + '_green_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_lbp(I[:, :, 2], Isegmented, options)
    Xntmp = [name + '_blue_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_lbp(rgb2gray(I), Isegmented, options)
    Xntmp = [name + '_gray_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_lbp(Ilab[:, :, 0], Isegmented, options)
    Xntmp = [name + '_L*_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_lbp(Ilab[:, :, 1], Isegmented, options)
    Xntmp = [name + '_A*_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    Xtmp, Xntmp = Bfx_lbp(Ilab[:, :, 2], Isegmented, options)
    Xntmp = [name + '_B*_normal' for name in Xntmp]

    Xstack.extend(Xtmp[0])
    Xnstack.extend(Xntmp)

    # Multiblock_LBP - RGB - normal
    ILabel = label(Isegmented)
    for region in regionprops(ILabel):
        minr, minc, maxr, maxc = region.bbox

    Xtmp = multiblock_lbp(I[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_red_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_green_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_blue_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - Gray - normal
    Xtmp = multiblock_lbp(rgb2gray(I), minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_gray_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - LBP - normal
    Xtmp = multiblock_lbp(Ilab[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_L*_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_A*_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_B*_normal'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - RGB - 0.5
    ILabel = label(Isegmented05)
    for region in regionprops(ILabel):
        minr, minc, maxr, maxc = region.bbox

    Xtmp = multiblock_lbp(I05[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_red_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I05[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_green_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I05[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_blue_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - Gray - normal
    Xtmp = multiblock_lbp(rgb2gray(I05), minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_gray_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - LBP - normal
    Xtmp = multiblock_lbp(Ilab05[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_L*_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab05[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_A*_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab05[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_B*_0.5'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - RGB - 0.25
    ILabel = label(Isegmented025)
    for region in regionprops(ILabel):
        minr, minc, maxr, maxc = region.bbox

    Xtmp = multiblock_lbp(I025[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_red_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I025[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_green_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I025[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_blue_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - Gray - normal
    Xtmp = multiblock_lbp(rgb2gray(I025), minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_gray_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - LBP - normal
    Xtmp = multiblock_lbp(Ilab025[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_L*_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab025[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_A*_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab025[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_B*_0.25'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - RGB - 0.125
    ILabel = label(Isegmented0125)
    for region in regionprops(ILabel):
        minr, minc, maxr, maxc = region.bbox

    Xtmp = multiblock_lbp(I0125[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_red_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I0125[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_green_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(I0125[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_blue_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - Gray - normal
    Xtmp = multiblock_lbp(rgb2gray(I0125), minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_gray_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    # Multiblock_LBP - LBP - normal
    Xtmp = multiblock_lbp(Ilab0125[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_L*_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab0125[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_A*_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    Xtmp = multiblock_lbp(Ilab0125[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
    Xntmp = 'multiblock_lbp_B*_0.125'

    Xstack.extend([Xtmp])
    Xnstack.extend([Xntmp])

    return Xstack, Xnstack
