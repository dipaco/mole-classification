# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import entropy
from skimage.color import rgb2gray
from matplotlib.pyplot import imshow, show


def entropy_rag(graph, labels, image,
                extra_arguments=[],
                extra_keywords={}):
    Gray = rgb2gray(image.astype(float))
    for n in graph:
        R = labels == n
        ii, jj = np.where(R)
        pixels = Gray[ii, jj].ravel()
        hist, _ = np.histogram(pixels, bins=np.arange(-0.5, 256))
        if pixels.size == 0:
            ent = np.float64(0.0)
        else:
            ent = entropy(hist)

        if np.isinf(ent) or np.isneginf(ent):
            ent = np.float64(0.0)

        #print 'nodo ', n, ' entr: ', ent

        graph.node[n].update({'labels': [n],
                              'pixels': pixels,
                              'entropy': ent})

    for x, y, d in graph.edges_iter(data=True):
        d['weight'] = mutual_information(graph.node[x]['pixels'],
                                         graph.node[y]['pixels'],
                                         graph.node[x]['entropy'],
                                         graph.node[y]['entropy'])


def _weight_entropy(graph, src, dst, n):
    return mutual_information(graph.node[dst]['pixels'],
                              graph.node[n]['pixels'],
                              graph.node[dst]['entropy'],
                              graph.node[n]['entropy'])

def mutual_information(pixelsA, pixelsB, HA, HB):
    if pixelsA.size + pixelsB.size == 0:
        HAB = 0
    else:
        hist, _ = np.histogram(np.hstack((pixelsA, pixelsB)), bins=np.arange(-0.5, 256))
        HAB = entropy(hist)

    if np.isinf(HAB) or np.isneginf(HAB):
        ent = np.float64(0.0)

    a = float(len(pixelsA))
    b = float(len(pixelsB))
    ab = a + b
    a /= ab
    b /= ab
    ans = HAB - (a * HA + b * HB)
    #print ' MI: ', ans
    if np.isinf(ans) or np.isneginf(ans):
        return np.float64(0.0)
    return ans


def merge_entropy(graph, src, dst, image, labels):
    graph.node[dst]['pixels'] = np.hstack((graph.node[dst]['pixels'], graph.node[src]['pixels']))
    if graph.node[dst]['pixels'].size == 0:
        graph.node[dst]['entropy'] = 0
    else:
        hist, _ = np.histogram(graph.node[dst]['pixels'], bins=np.arange(-0.5, 256))
        graph.node[dst]['entropy'] = entropy(hist)
