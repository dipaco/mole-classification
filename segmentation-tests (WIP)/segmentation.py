# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import entropy
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import threshold_otsu
from skimage.morphology import label, closing, disk, dilation
from future import graph
from balu.FeatureExtraction import Bfx_haralick
from balu.FeatureAnalysis import Bfa_jfisher
from utils import compare_jaccard
from matplotlib.pyplot import imshow, show, figure, subplot


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
        options = {'dharalick': 3}
        R = labels == n
        bbox = all_rp[n]['bbox']
        Xhtmp, _ = Bfx_haralick(
            Gray[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1],
            R[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1],
            options)

        graph.node[n].update({'labels': [n],
                              'haralick': Xhtmp,
                              'mask': R,
                              'image': Gray})

    for x, y, d in graph.edges_iter(data=True):
        nodex = graph.node[x]['haralick']
        nodey = graph.node[y]['haralick']

        # x[numpy.isneginf(x)] = 0
        d['weight'] = np.linalg.norm(nodex.T - nodey.T)


def merge_haralick(graph, src, dst, image, labels):
    options = {'dharalick': 3}
    mask = np.logical_or(graph.node[dst]['mask'], graph.node[src]['mask'])
    bbox = regionprops(mask.astype(int))[0]['bbox']
    image = graph.node[dst]['image']
    Xhtmp, _ = Bfx_haralick(
        image[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1],
        mask[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1],
        options)
    graph.node[dst]['haralick'] = Xhtmp
    graph.node[dst]['mask'] = mask


def _weight_haralick(graph, src, dst, n):
    nodedst = graph.node[dst]['haralick']
    noden = graph.node[n]['haralick']
    dd = np.linalg.norm(nodedst.T - noden.T)
    return dd


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


def segment(I, mask, method):

    # n_segments sujeto a cambios para optimización de la segmentación
    L = slic(I, n_segments=400)
    Islic = mark_boundaries(I, L)

    if method == 'color':
        g = graph.rag_mean_color(I, L)
        # lc = graph.draw_rag(L, g, Islic)
        L2 = graph.merge_hierarchical(L, g, thresh=50, rag_copy=False,
                                      in_place_merge=True,
                                      merge_func=merge_mean_color,
                                      weight_func=_weight_mean_color)
    elif method == 'entropy':
        g = graph.region_adjacency_graph(L, image=I, describe_func=entropy_rag)
        L2 = graph.merge_hierarchical(L, g, thresh=0.3, rag_copy=False,
                                      in_place_merge=True,
                                      merge_func=merge_entropy,
                                      weight_func=_weight_entropy,
                                      image=I)
    elif method == 'haralick':
        g = graph.region_adjacency_graph(L, image=I, describe_func=haralick_rag)
        L2 = graph.merge_hierarchical(L, g, thresh=150, rag_copy=False,
                                      in_place_merge=True,
                                      merge_func=merge_haralick,
                                      weight_func=_weight_haralick,
                                      image=I)

    Islic2 = mark_boundaries(I, L2)

    s = np.zeros((I.shape[0:2]), dtype=np.uint8)
    L2label = label(L2)

    IGray = rgb2gray(I)  # * mask
    thresh = threshold_otsu(IGray)
    IOtsu = IGray <= thresh
    IOtsu = np.logical_and(IOtsu, mask)
    # IOtsu = closing(IOtsu, selem=disk(5))

    # Islic3 = mark_boundaries(IOtsu, L2)
    # g3 = graph.rag_mean_color(I, L2)
    # lc2 = graph.draw_rag(L2, g3, Islic3, border_color='#ff6600')
    # imshow(lc2, cmap='gray')
    # show()

    J = np.zeros(L2label.max() + 1)
    for i in range(0, L2label.max() + 1):
        lbl = np.logical_and((L2label == i), mask)
        lbl = binary_fill_holes(lbl)
        jaccard = compare_jaccard(IOtsu, lbl)
        J[i] = jaccard

    sMask = np.logical_and((L2label == np.argmax(J)), mask)
    sMask = closing(sMask, selem=disk(3))
    slabel = label(sMask)

    # Calculates the area of each label to select the one with the
    # máximum área
    max = 0
    iMax = -1
    for i in range(1, slabel.max() + 1):
        if np.sum(slabel == i) > max:
            max = np.sum(slabel == i)
            iMax = i

    Isegmented = slabel == iMax
    Isegmented = binary_fill_holes(Isegmented)
    Isegmented = dilation(Isegmented, selem=disk(8))
    return Isegmented, Islic, Islic2