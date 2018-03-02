# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import entropy
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.segmentation import slic, mark_boundaries, relabel_sequential
from skimage.filters import threshold_otsu
from skimage.morphology import label, disk, dilation, erosion, square, remove_small_objects, opening
from skimage.feature import canny
from skimage.exposure import equalize_hist, equalize_adapthist
from future import graph
from utils import compare_jaccard


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
            ent = entropy(hist, qk=None)

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


def segment(I, mask, method, k=400):

    # k_segments
    L = slic(I, n_segments=k) * mask
    Islic = mark_boundaries(I, L)

    '''RECOMENDED THRESHOLDS BY METHOD:
        color_thresh = 1000
        entropy_thresh = 0.6
        haralick_thres = 300
    '''

    # Merge superpixels until get just 2 (lession and background) or until
    # convergence criterion is reached
    min_size_object = 0.02 * mask.size
    merge_thresh = find_merge_threshold(I, L, mask, upper_thresh=1000, epsilon=0.0001, max_iter=100, method=method, min_size_object=min_size_object)
    L2 = merge_superpixels(I, L, mask, merge_thresh, method, min_size_object)

    Islic2 = mark_boundaries(I, L2)

    s = np.zeros((I.shape[0:2]), dtype=np.uint8)
    L2label = label(L2)

    IGray = rgb2gray(I)  # * mask
    IGray = equalize_adapthist(IGray)
    thresh = threshold_otsu(IGray)
    IOtsu = IGray <= thresh
    IOtsu = np.logical_and(IOtsu, mask)

    Isegmented = max_jaccard_criterion(IOtsu, L2label, mask)
    return Isegmented, L2, Islic2, IOtsu, Islic


def find_merge_threshold(I, L, mask, upper_thresh, epsilon, max_iter, method, min_size_object):
    a = 0
    b = upper_thresh
    iter = 0
    merge_thresh = -50
    optimal_thresh = (a + b) / 2.0
    while True:
        iter += 1
        prev_thresh = merge_thresh
        merge_thresh = (a + b) / 2.0

        L2 = merge_superpixels(I, L, mask, merge_thresh, method, min_size_object)

        print('it:', iter, ' T: ', a, b, merge_thresh, 'L: ', L2.max())

        '''imshow(L2)
        colorbar()
        show()'''

        if L2.max() < 2:
            b = merge_thresh
        elif L2.max() > 2:
            optimal_thresh = merge_thresh
            a = merge_thresh
        else:
            optimal_thresh = merge_thresh
            break

        # Stop criterion by max iteration
        if iter >= max_iter:
            break

        # Stop criterion by threshold delta
        if np.abs(prev_thresh - merge_thresh) < epsilon:
            break

    return optimal_thresh


def merge_superpixels(I, L, mask, merge_thresh, method, min_size_object):
    if method == 'color':
        g = graph.rag_mean_color(I, L)
        # lc = graph.draw_rag(L, g, Islic)
        L2 = graph.merge_hierarchical(L, g, thresh=merge_thresh, rag_copy=False,
                                      in_place_merge=True,
                                      merge_func=merge_mean_color,
                                      weight_func=_weight_mean_color)
    elif method == 'entropy':
        g = graph.region_adjacency_graph(L, image=I, describe_func=entropy_rag)
        L2 = graph.merge_hierarchical(L, g, thresh=merge_thresh, rag_copy=False,
                                      in_place_merge=True,
                                      merge_func=merge_entropy,
                                      weight_func=_weight_entropy,
                                      image=I)

    # Deletes superpixels outside the mask, removes small objects
    # and relabel de superpixels
    L2 *= mask
    L2 = remove_small_objects(L2, min_size=min_size_object)
    L2, _, _ = relabel_sequential(L2)
    return L2


def coequialization_saliency(I, labels, mask):

    sal = np.zeros_like(labels)
    for i in range(labels.max() + 1):
        for j in range(i + 1, labels.max() + 1):
            A = I * (np.logical_or(labels == i, labels == j))
            A = equalize_hist(A)
            if A.sum() <= 0 or (A.max() == A.min()):
                continue
            A = A < threshold_otsu(A)
            sal += A.astype(int)

    sal = sal.astype(float) / (labels.max()*(labels.max() + 1)/2.0) * mask

    Isegmented = np.zeros(labels.shape, dtype=bool)
    for i in range(labels.max() + 1):
        aa = np.mean(sal * (labels == i))
        if aa > 0.5:
            Isegmented = np.logical_or(Isegmented, labels == i)

    return Isegmented

def edge_support_criterion(edges, labels):

    F = np.zeros_like(edges)
    for i in range(0, labels.max() + 1):
        l = binary_fill_holes(labels == i)
        perimeter = l - erosion(l, square(3))
        '''imshow(l, cmap='gray')
        figure()
        imshow(perimeter, cmap='gray')
        figure()
        imshow(np.logical_and(perimeter, edges), cmap='gray')
        show()'''
        c = np.logical_and(perimeter, edges).sum() / float(perimeter.sum())
        if c > 0.5:
            F = np.logical_or(F, l)

        #F = binary_fill_holes(F)
        #F = dilation(F, selem=disk(8))
    return F


def max_jaccard_criterion(IOtsu, L2label, mask):
    J = np.zeros(L2label.max() + 1)
    for i in range(1, L2label.max() + 1):
        lbl = np.logical_and((L2label == i), mask)
        jaccard = compare_jaccard(IOtsu, lbl)
        J[i] = jaccard
    sMask = np.logical_and((L2label == np.argmax(J)), mask)
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
    Isegmented = opening(Isegmented, selem=disk(8))
    Isegmented = dilation(Isegmented, selem=disk(8))
    return Isegmented

'''
def segmentYCrCb(I, mask, method):
    YCrCb = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)

    ii, jj = np.where(np.logical_and(YCrCb[:, :, 1] > 133,
                                     np.logical_and(YCrCb[:, :, 1] < 173,
                                                    np.logical_and(YCrCb[:, :, 2] > 77, YCrCb[:, :, 2] < 127))))

    Isegmented = np.ones(I.shape[0:2]).astype(bool)
    Isegmented[ii, jj] = False

    return Isegmented, Isegmented, Isegmented, Isegmented
'''