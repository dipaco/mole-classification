# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import entropy
from skimage.color import rgb2gray
from skimage.measure import regionprops
from balu.FeatureExtraction import Bfx_haralick
from balu.FeatureAnalysis import Bfa_jfisher
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


def haralick_rag(graph, labels, image,
                 extra_arguments=[],
                 extra_keywords={}):
    all_rp = regionprops(labels + 1)
    Gray = rgb2gray(image.astype(float))
    for n in graph:
        Xhstack = []
        options = {'dharalick': 3}

        '''
        J = image[:, :, 0]
        Xhtmp, _ = Bfx_haralick(J, labels == n, options)

        Xhstack.extend(Xhtmp[0])

        J = image[:, :, 1]
        Xhtmp, _ = Bfx_haralick(J, labels == n, options)

        Xhstack.extend(Xhtmp[0])

        J = image[:, :, 2]
        Xhtmp, _ = Bfx_haralick(J, labels == n, options)

        Xhstack.extend(Xhtmp[0])

        '''
        R = labels == n
        bbox = all_rp[0]['bbox']
        Xhtmp, _ = Bfx_haralick(
            Gray[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1],
            R[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1],
            options)
        Xhstack.extend(Xhtmp[0])

        graph.node[n].update({'labels': [n],
                              'haralick': Xhstack})

    for x, y, d in graph.edges_iter(data=True):
        dh = []
        Xh = []

        nodex = graph.node[x]['haralick']
        Xh.append(nodex)
        dh.extend([0])

        nodey = graph.node[y]['haralick']
        Xh.append(nodey)
        dh.extend([1])

        # TODO: Cambiar esto por la distancia euclideana
        # TODO: No utilizar listas sino siempre arrays de numpy
        # x[numpy.isneginf(x)] = 0
        Xh = np.asarray(Xh)
        d['weight'] = np.linalg.norm(Xh[0, :].T - Xh[1, :].T)
        # d['weight'] = Bfa_jfisher(np.asarray(Xh), np.asarray(dh))
        # print d['weight']


def merge_haralick(graph, src, dst, image, labels):
    options = {'dharalick': 3}
    Xh = []
    Xhtmp, _ = Bfx_haralick(rgb2gray(image), (labels == dst) + (labels == src), options)
    Xh.extend(Xhtmp[0])
    graph.node[dst]['haralick'] = Xh


def _weight_haralick(graph, src, dst, n):
    dh = []
    Xh = []

    nodedst = graph.node[dst]['haralick']
    Xh.append(nodedst)
    dh.extend([0])

    noden = graph.node[n]['haralick']
    Xh.append(noden)
    dh.extend([1])

    return Bfa_jfisher(np.asarray(Xh), np.asarray(dh))


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.
    The method expects that the mean color of `dst` is already computed.
    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.
    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_mean_color(graph, src, dst, image, labels):
    """Callback called before merging two nodes of a mean color distance graph.
    This method computes the mean color of `dst`.
    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])