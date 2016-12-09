# -*- coding: utf-8 -*-
import numpy as np
from skimage.draw import ellipse


def compare_jaccard(img1, img2):
    num = np.sum(np.logical_and(img1, img2))
    den = float(np.sum(np.logical_or(img1, img2)))
    if den == 0.0:
        jaccard = 0.0
    else:
        jaccard = num / den

    return jaccard


def get_mask(s):
    mask = np.zeros(s, dtype=np.uint8)
    rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 2) - 1, round(s[1] / 2) - 1)
    mask[rr, cc] = 1
    return mask