'''
Requeriments:
- Python (We use 3.5).
- Packages: matplotlib, skimage, numpy, mahotas.
'''

import numpy as np
from matplotlib.image import imread
from skimage.color import rgb2gray, label2rgb, gray2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import slic, mark_boundaries
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle
from skimage.future import graph
from matplotlib import colors
from skimage.draw import ellipse
from skimage.measure import label, compare_mse, compare_ssim, compare_psnr
from mahotas.features import haralick
#conda install -c https://conda.anaconda.org/conda-forge mahotas
from balu.FeatureAnalysis import Bfa_jfisher
from skimage.filters import gaussian
from skimage.morphology import opening, disk, closing
import os
import fnmatch
from scipy.ndimage.morphology import binary_fill_holes
from balu.FeatureExtraction import Bfx_haralick, Bfx_geo, Bfx_basicgeo

def magic(segmentationProcess=True, saveSegmentation=True, featuresProcess=True):
    global path, pathSegmentation

    if saveSegmentation:
        counter = 0
        if not os.path.exists(pathSegmentation):
            os.makedirs(pathSegmentation)

    def compare_jona(img1, img2):
        c = 0
        for i in range(len(GT)):
            for j in range(len(GT[i])):
                if GT[i][j] == Isegmented[i][j]:
                    c += 1
        return c / (len(GT) * len(GT[0])), 0.58

    def _weight_haralick(graph, src, dst, n):
        """Callback to handle merging nodes by haralick method.
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
        s = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
        for i in graph.node[dst]['labels']:
            s += L == i

        hdst = haralick(gray2rgb(s) * I)

        s = np.ones((IOriginal.shape[0:2]), dtype=np.uint8)
        for i in graph.node[n]['labels']:
            s += L == i
        hn = haralick(gray2rgb(s) * I)

        #print(hdst)
        #print(hn)

        J = Bfa_jfisher(hdst + hn, np.zeros(len(hdst)) + np.ones(len(hn)))

        #print(J)

        #diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        #diff = np.linalg.norm(diff)
        #return {'weight': diff}
        return J

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

    for image in fnmatch.filter(os.listdir('imgs'), '*.bmp'):

        if segmentationProcess:
            IOriginal = imread(path + '/' + image)

            mask = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
            rr, cc = ellipse(round(IOriginal.shape[0]/2), round(IOriginal.shape[1]/2), round(IOriginal.shape[0]/2)-1, round(IOriginal.shape[1]/2)-1)
            mask[rr, cc] = 1

            maskrgb = gray2rgb(mask)
            #imshow(mask)
            #show()

            #IOriginalGray = rgb2gray(IOriginal)

            I = maskrgb * IOriginal

            #imshow(I)
            #show()

            GT = rgb2gray(imread(path + 'GT/' + image[:-4] + '_lesion.bmp').astype(float)) > 120

            # n_segments sujeto a cambios para optimización de la segmentación
            L = slic(I, n_segments=400)

            #subplot(1, 2, 1)
            Islic = mark_boundaries(I, L)
            #imshow(Islic)
            #show()
            #subplot(1, 2, 2)
            #imshow(mark_boundaries(GT, L, color=(1, 0, 0)))
            #show()

            g = graph.rag_mean_color(I, L)
            lc = graph.draw_rag(L, g, Islic)

            #imshow(lc)
            #show()

            L2 = graph.merge_hierarchical(L, g, thresh=50, rag_copy=False,
                                          in_place_merge=True,
                                          merge_func=merge_mean_color,
                                          weight_func=_weight_mean_color)

            Islic2 = mark_boundaries(I, L2)
            #figure()
            g2 = graph.rag_mean_color(I, L2)
            lc2 = graph.draw_rag(L2, g2, Islic2)

            '''
            out = label2rgb(L2, I, kind='avg')
            out = mark_boundaries(out, L2, (0, 0, 0))
            '''

            #subplot(1, 2, 1)
            #imshow(GT, cmap='gray')

            #subplot(1, 2, 2)
            #imshow(lc2)
            #show()

            s = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
            L2label = label(L2)

            IGray = rgb2gray(IOriginal)# * mask
            IGaussian = gaussian(IGray, sigma=0.5)
            thresh = threshold_otsu(IGray)
            IOtsu = IGray <= thresh

            #Islic3 = mark_boundaries(IOtsu, L2)
            #g3 = graph.rag_mean_color(I, L2)
            #lc2 = graph.draw_rag(L2, g3, Islic3, border_color='#ff6600')
            #imshow(lc2, cmap='gray')
            #show()

            for i in range(1, L2label.max()):
                lbl = (L2label == i)
                countc = 0
                count = 0
                for j in range(len(lbl)):
                    for k in range(len(lbl[i])):
                        if not lbl[j][k] == 0:
                            if IOtsu[j][k]:
                                countc += 1
                            count += 1
                if (countc/count) >= 0.4:
                    s += lbl

            sMask = s * mask
            sMaskClose = closing(sMask, selem=disk(3))
            sMaskOpen = opening(sMaskClose, selem=disk(3))  # ??

            slabel = label(sMaskOpen)

            Isegmented = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
            max = 0
            iMax = 0
            for i in range(1, slabel.max()):
                if np.sum(slabel == i) > max:
                    max = np.sum(slabel == i)
                    iMax = i

            Isegmented = slabel == iMax

            if Isegmented[0][0] and Isegmented[0][len(Isegmented[0])-1] and Isegmented[len(Isegmented)-1][0] and Isegmented[len(Isegmented)-1][len(Isegmented[0])-1]:
                Isegmented = np.invert(Isegmented)

            Isegmented = binary_fill_holes(Isegmented)

            #subplot(1, 2, 1)
            #imshow(GT, cmap='gray')

            #subplot(1, 2, 2)
            #imshow(Isegmented, cmap='gray')
            #show()

            if saveSegmentation:
                imsave(pathSegmentation + '/' + image[:-4] + '_our.png', Isegmented, cmap='gray')
                counter += 1
                print(counter, '/', len(fnmatch.filter(os.listdir('imgs'), '*.bmp')))

            '''
            s = np.ones((IOriginal.shape[0:2]), dtype=np.uint8)
            L2label = label(L2)

            for i in range(1, L2label.max()):
                R = 0
                G = 0
                B = 0
                count = 0
                It = (gray2rgb(L2label == i) * I)
                for j in range(len(I)):
                    for k in range(len(I[j])):
                        if It[j][k][0] > 0 and It[j][k][1] > 0 and It[j][k][2] > 0 and:              #??
                            print('R +=', It[j][k][0])
                            R += It[j][k][0]
                            print('G +=', It[j][k][1])
                            G += It[j][k][1]
                            print('B +=', It[j][k][2])
                            B += It[j][k][2]
                            count += 1
                print(R/count)
                print(G/count)
                print(B/count)
                imshow(gray2rgb(L2label == i) * I)
                show()
            '''
        if not segmentationProcess:
            IOriginal = imread(path + '/' + image)

            mask = np.zeros((IOriginal.shape[0:2]), dtype=np.uint8)
            rr, cc = ellipse(round(IOriginal.shape[0] / 2), round(IOriginal.shape[1] / 2),
                             round(IOriginal.shape[0] / 2) - 1, round(IOriginal.shape[1] / 2) - 1)
            mask[rr, cc] = 1

            maskrgb = gray2rgb(mask)

            I = maskrgb * IOriginal

            GT = (rgb2gray(imread(path + 'GT/' + image[:-4] + '_lesion.bmp').astype(float)) * mask) > 120

            Isegmented255 = rgb2gray(imread(pathSegmentation + '/' + image[:-4] + '_our.png').astype(float))
            Isegmented = Isegmented255 > 120

        if featuresProcess:
            """
            TO-DO:
                - Geo                   *
                - hu                    *
                - flusser               *
                - furier                *
                - Haralick
                - Entropia
                - Media
                - Algo más
                - Otra cosa
                - Y ajá
                - gabor


                - Eliminar cositas constantes
                - Feature selection para sacar las mejores
            """

            '''
            ICharacteristics = gray2rgb(Isegmented) * I

            hara = haralick(ICharacteristics, ignore_zeros=True)
            haralick_labels = ["Angular Second Moment",
                               "Contrast",
                               "Correlation",
                               "Sum of Squares: Variance",
                               "Inverse Difference Moment",
                               "Sum Average",
                               "Sum Variance",
                               "Sum Entropy",
                               "Entropy",
                               "Difference Variance",
                               "Difference Entropy",
                               "Information Measure of Correlation 1",
                               "Information Measure of Correlation 2",
                               "Maximal Correlation Coefficient"]
            print(hara)
            print(haralick_labels)
            '''

            '''
            IGreenChannel = I[:, :, 1]
            X, Xn = Bfx_haralick(IGreenChannel, Isegmented, options={'dharalick': 3})
            print(X)
            print(Xn)
            '''

            #print('mse ', image[:-4], compare_mse(GT, Isegmented))
            #mean_mse += compare_mse(GT, Isegmented)
            #print('ssim', image[:-4], compare_ssim(GT, Isegmented))
            #mean_ssim += compare_ssim(GT, Isegmented)
            #print('pnsr', image[:-4], compare_psnr(GT, Isegmented))
            #mean_pnsr += compare_psnr(GT, Isegmented)
            #print('jona', image[:-4], compare_jona(GT, Isegmented))
            #mean_jona += compare_jona(GT, Isegmented)
            #print('')

            jona, mean_jona = compare_jona(GT, Isegmented)

            if (jona >= mean_jona) and (np.sum(Isegmented255) > 0):
                options = {'b': [
                    {'name': 'basicgeo', 'options': {'show': False}},                       # basic geometric features
                    {'name': 'hugeo', 'options': {'show': False}},                          # Hu moments
                    {'name': 'flusser', 'options': {'show': False}},                        # Flusser moments
                    {'name': 'fourierdes', 'options': {'show': False, 'Nfourierdes': 12}}   # Fourier descriptors
                ]}

                X, Xn = Bfx_geo(Isegmented255, options)
                print(X)
                print(Xn)




path = 'imgs'
pathSegmentation = 'our_segmentation'
magic(False, False, True)

