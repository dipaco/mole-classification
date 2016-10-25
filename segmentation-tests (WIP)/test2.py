import numpy as np
from matplotlib.image import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import slic, mark_boundaries
from matplotlib.pyplot import show, imshow, subplot
from skimage.future import graph
from matplotlib import colors

I = imread('imgs/IMD242.bmp')
GT = rgb2gray(imread('imgs/IMD242_lesion.bmp').astype(float)) > 120

# Try other numbers
L = slic(I, n_segments=900)

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
