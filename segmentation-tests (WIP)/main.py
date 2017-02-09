# -*- coding: utf-8 -*-
'''
Requeriments:
- Python (We use 3.5).
- Packages: matplotlib, skimage, numpy, mahotas.
'''

import numpy as np
from PH2Dataset import PH2Dataset
from matplotlib.image import imread
from skimage.color import rgb2gray, label2rgb, gray2rgb
from skimage.segmentation import slic, mark_boundaries
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle, colorbar
from skimage.measure import label, compare_mse
from balu.DataSelectionAndGeneration import Bds_nostratify
import os
import fnmatch
from balu.FeatureExtraction import Bfx_haralick, Bfx_geo, Bfx_basicgeo, Bfx_lbp
from skimage.exposure import histogram
from scipy.special import entr
from scipy.io import savemat, loadmat
from balu.FeatureSelection import Bfs_clean, Bfs_sfs
from balu.Classification import Bcl_structure
from balu.PerformanceEvaluation import Bev_performance, Bev_confusion

#Segmentation
from segmentation import segment
from utils import get_mask, compare_jaccard


def magic(imgPath, imgSegPath, method='color', segmentationProcess=True, featuresProcess=True, trainAndTest=True):
    """
    :param imgPath:
    :param imgSegPath:
    :param method: color | entropy | haralick
    :param segmentationProcess:
    :param featuresProcess:
    :param trainAndTest:
    :return:
    """

    path = imgPath
    pathSegmentation = imgSegPath
    all_mse = []
    all_jaccard = []

    if featuresProcess:
        X = []
        Xn = []
        d = []
        imagesNames = []

    #Set a class to manage the whole dataset
    dataset = PH2Dataset('PH2Dataset') #recibe la carpeta donde está el dataset (ruta)
    #dataset.set_sample(percentage=0.05)
    #dataset.set_sample(image_indices=[0, 50, 2, 5, 198])
    dataset.set_sample(image_names=['IMD204', 'IMD380', 'IMD135', 'IMD408', 'IMD003', 'IMD306', 'IMD080', 'IMD035', 'IMD103'])
    dataset.exclude_from_sample(image_names=['IMD417'])

    if segmentationProcess or featuresProcess:
        print("{:10} {:20} {:20}".format('Imagen', 'MSE', 'JACCARD'))
        for image_idx in range(dataset.num_images):

            image = dataset.image_names[image_idx]

            if segmentationProcess:
                # reads the image information from the dataset
                IOriginal = dataset.get_image_data(image_idx)

                # Gets the mask to avoid dark areas in segmentation
                mask = get_mask(IOriginal.shape[0:2])
                I = gray2rgb(mask) * IOriginal
                GT = (rgb2gray(dataset.get_ground_truth_data(image_idx).astype(float)) * mask) > 120

                #Segment the each mole
                print('Segmenting image {0} ({1} / {2})'.format(dataset.image_names[image_idx], image_idx + 1, dataset.num_images))
                Isegmented = segment(I, mask, method=method)

                auxmse = compare_mse(GT, Isegmented)
                all_mse.append(auxmse)
                auxjacc = compare_jaccard(GT, Isegmented)
                all_jaccard.append(auxjacc)

                print("{:10} {:0.25f} {:0.25f}".format(image, auxmse, auxjacc))

                if not os.path.exists(pathSegmentation):
                    os.makedirs(pathSegmentation)

                imsave(pathSegmentation + '/' + image + '_our.png', Isegmented.astype(int), cmap='gray')

            else: #SEGMENTATION IS DONE AND SAVED
                # reads the image information from the dataset
                IOriginal = dataset.get_image_data(image_idx)
                #Gets the mask to avoid dark areas in segmentation
                mask = get_mask(IOriginal.shape[0:2])
                I = gray2rgb(mask) * IOriginal
                GT = (rgb2gray(dataset.get_ground_truth_data(image_idx).astype(float)) * mask) > 120
                Isegmented = rgb2gray(imread(pathSegmentation + '/' + image + '_our.png').astype(float))

            if featuresProcess:
                if np.sum(Isegmented) > 0:
                    print('Extracting feature to image {0} ({1} / {2})'.format(dataset.image_names[image_idx], image_idx + 1,
                                                                    dataset.num_images))
                    Xstack = []
                    Xnstack = []

                    options = {'b': [
                        {'name': 'basicgeo', 'options': {'show': False}},                       # basic geometric features
                        {'name': 'hugeo', 'options': {'show': False}},                          # Hu moments
                        {'name': 'flusser', 'options': {'show': False}},                        # Flusser moments
                        {'name': 'fourierdes', 'options': {'show': False, 'Nfourierdes': 12}},  # Fourier descriptors
                    ]}

                    Xtmp, Xntmp = Bfx_geo(Isegmented, options)
                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    options = {'dharalick': 3}  # 3 pixels distance for coocurrence

                    J = rgb2gray(I.astype(float))
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented, options)  # Haralick features
                    Xntmp = [name + '_gray' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    options = {
                        'weight': 0,  # Weigth of the histogram bins
                        'vdiv': 1,  # one vertical divition
                        'hdiv': 1,  # one horizontal divition
                        'samples': 8,  # number of neighbor samples
                        'mappingtype': 'nri_uniform'  # uniform LBP
                    }

                    a, _ = histogram(np.round(rgb2gray(I.astype(float))))
                    a /= a.sum()

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

                    X.append(Xstack)
                    if len(Xn) == 0:
                        Xn = Xnstack

                    d.extend([dataset.get_image_class(image_idx)])
                    imagesNames.extend([image])

        if featuresProcess:
            print(X)
            print(Xn)
            print(d)
            print(imagesNames)
            d = np.array(d)
            savemat('X-Xn-d-names.mat', {'X': X, 'Xn': Xn, 'd': d, 'imagesNames': imagesNames})

        if segmentationProcess:
            print("{:10} {:20} {:20}".format('Indice', 'Media', 'Desviacion'))
            print("{:10} {:0.20f} {:0.20f}".format('MSE', sum(all_mse) / len(all_mse), np.std(all_mse)))
            print("{:10} {:0.20f} {:0.20f}".format('JACCARD', sum(all_jaccard) / len(all_jaccard), np.std(all_jaccard)))

    if trainAndTest:
        data = loadmat('X-Xn-d-names.mat')
        X = data['X']
        Xn = data['Xn']
        d = data['d'][0]
        imagesNames = data['imagesNames']

        # training
        print('training')
        sclean = Bfs_clean(X, 1)
        Xclean = X[:, sclean]
        Xnclean = Xn[sclean]
        Xtrain, dtrain, Xtest, dtest = Bds_nostratify(Xclean, d, 0.9)


        b = [
            {'name': 'lda', 'options': {'p': []}},
            {'name': 'maha', 'options': {}},
            {'name': 'qda', 'options': {'p': []}},
            {'name': 'svm', 'options': {'kernel': 1}},
            {'name': 'svm', 'options': {'kernel': 2}},
            {'name': 'knn', 'options': {'k': 5}},
            {'name': 'nn', 'options': {'method': 1, 'iter': 15}}
            {'name': 'nn', 'options': {'method': 1, 'iter': 15}}
        ]
        op = b
        struct = Bcl_structure(Xtrain, dtrain, op)

        print('training')
        ds, _ = Bcl_structure(Xtest, struct)
        for i in range(len(op)):
            T, p = Bev_confusion(dtest, ds[:, i])
            print(b[i]['name'], ': ', p)
            #print(T)

'''
Clinical Diagnosis:
    1 - Common Nevus;
    2 - Atypical Nevus;
    3 - Melanoma.
'''

path = 'imgs'
pathSegmentation = 'our_segmentation'
magic(imgPath=path,
      imgSegPath=pathSegmentation,
      method='color',
      segmentationProcess=False,
      featuresProcess=False,
      trainAndTest=True)
