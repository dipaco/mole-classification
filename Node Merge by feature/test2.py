import numpy as np
from matplotlib.image import imread
from skimage.color import rgb2gray , label2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import slic, mark_boundaries
from matplotlib.pyplot import show, imshow, subplot, figure
from skimage.future import graph
from matplotlib import colors

I = imread('imgs/IMD242.bmp')
GT = rgb2gray(imread('imgs/IMD242_lesion.bmp').astype(float)) > 120

#n_segments sujeto a cambios para optimización de la segmentación
L = slic(I, n_segments=400)

subplot(1, 2, 1)
Islic = mark_boundaries(I, L)
imshow(Islic)

subplot(1, 2, 2)
imshow(mark_boundaries(GT, L, color=(1, 0, 0)))
show()


g = graph.rag_mean_color(I, L)
lc = graph.draw_rag(L, g, Islic)

imshow(lc)
show()

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
    #return {'weight': diff}
    return diff

def merge_mean_color(graph, src, dst):
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



L2 = graph.merge_hierarchical(L, g, thresh=50, rag_copy=False,
                              in_place_merge=True,
                              merge_func=merge_mean_color,
                              weight_func=_weight_mean_color)

Islic2 = mark_boundaries(I, L2)
figure()
g2 = graph.rag_mean_color(I, L2)
lc2 = graph.draw_rag(L2, g2, Islic2)

'''out = label2rgb(L2, I, kind='avg')
out = mark_boundaries(out, L2, (0, 0, 0))'''

subplot(1, 2, 1)
imshow(Islic)


subplot(1, 2, 2)
imshow(lc2)
show()
