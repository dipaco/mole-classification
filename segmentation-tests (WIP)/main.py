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
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle, colorbar, savefig
from skimage.measure import label, compare_mse, regionprops
from balu.DataSelectionAndGeneration import Bds_nostratify
import os
import fnmatch
from balu.FeatureExtraction import Bfx_haralick, Bfx_geo, Bfx_basicgeo, Bfx_lbp
from balu.InputOutput import Bio_plotfeatures
from skimage.exposure import histogram
from scipy.special import entr
from scipy.io import savemat, loadmat
from balu.FeatureSelection import Bfs_clean, Bfs_sfs
from balu.Classification import Bcl_structure
from balu.PerformanceEvaluation import Bev_performance, Bev_confusion
from skimage.feature import multiblock_lbp

#Segmentation
from segmentation import segment
from utils import get_mask, compare_jaccard


def magic(imgPath, imgResults, method='color', segmentationProcess=True, featuresProcess=True, trainAndTest=True):
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
    pathSegmentation = os.path.join(imgResults, 'our_segmentation')
    pathResults = os.path.join(imgResults, 'figures')
    all_mse = []
    all_jaccard = []

    if featuresProcess:
        X = []
        Xn = []
        d = []
        imagesNames = []

    #Set a class to manage the whole dataset
    dataset = PH2Dataset('PH2Dataset')
    #dataset.set_sample(percentage=0.1)
    #dataset.set_sample(image_indices=[0, 50, 2, 5, 198])
    #dataset.set_sample(image_names=['IMD155', 'IMD306', 'IMD382', 'IMD048', 'IMD347', 'IMD386', 'IMD103', 'IMD203', 'IMD312', 'IMD085', 'IMD424', 'IMD384', 'IMD037', 'IMD080', 'IMD369', 'IMD431', 'IMD339', 'IMD031', 'IMD108', 'IMD226'])
    #dataset.exclude_from_sample(image_names=['IMD417'])
    dataset.exclude_from_sample(image_names=['IMG006', 'IMD008', 'IMD009', 'IMD014', 'IMD019', 'IMD023', 'IMD024', 'IMD032', 'IMD033', 'IMD035', 'IMD037', 'IMD048', 'IMD049', 'IMD058', 'IMD061', 'IMD064', 'IMD085', 'IMD088', 'IMD090', 'IMD091', 'IMD101', 'IMD105', 'IMD112', 'IMD118', 'IMD126', 'IMD135', 'IMD137', 'IMD138', 'IMD147', 'IMD152', 'IMD153', 'IMD154', 'IMD155', 'IMD157', 'IMD159', 'IMD160', 'IMD166', 'IMD168', 'IMD170', 'IMD177', 'IMD182', 'IMD196', 'IMD198', 'IMD200', 'IMD207', 'IMD208', 'IMD219', 'IMD240', 'IMD251', 'IMD254', 'IMD278', 'IMD279', 'IMD280', 'IMD284', 'IMD304', 'IMD339', 'IMD349', 'IMD356', 'IMD360', 'IMD364', 'IMD367', 'IMD368', 'IMD371', 'IMD372', 'IMD375', 'IMD378', 'IMD381', 'IMD382', 'IMD388', 'IMD390', 'IMD397', 'IMD398', 'IMD400', 'IMD403', 'IMD404', 'IMD405', 'IMD406', 'IMD407', 'IMD408', 'IMD409', 'IMD410', 'IMD411', 'IMD413', 'IMD417', 'IMD419', 'IMD420', 'IMD421', 'IMD424', 'IMD425', 'IMD426', 'IMD427', 'IMD430', 'IMD431', 'IMD432', 'IMD433', 'IMD435', 'IMD436'])

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
                Isegmented, Islic, Islic2 = segment(I, mask, method=method)

                auxmse = compare_mse(GT, Isegmented)
                all_mse.append(auxmse)
                auxjacc = compare_jaccard(GT, Isegmented)
                all_jaccard.append(auxjacc)

                print("{:10} {:0.25f} {:0.25f}".format(image, auxmse, auxjacc))

                if not os.path.exists(pathSegmentation):
                    os.makedirs(pathSegmentation)
                if not os.path.exists(pathResults):
                    os.makedirs(pathResults)

                subplot(2, 3, 1)
                title('Original')
                imshow(IOriginal)
                subplot(2, 3, 2)
                title('Ground Truth')
                imshow(GT, cmap='gray')
                subplot(2, 3, 3)
                title('Our Segmentation')
                imshow(Isegmented, cmap='gray')
                subplot(2, 3, 4)
                title('SuperPixels')
                imshow(Islic)
                subplot(2, 3, 5)
                title('Merged SuperPixels')
                imshow(Islic2)
                savefig(pathResults + '/' + image + '_our.png')


                imsave(pathSegmentation + '/' + image + '_our.png', 255*Isegmented.astype(int), cmap='gray')

            else: #SEGMENTATION IS DONE AND SAVED
                # reads the image information from the dataset
                IOriginal = dataset.get_image_data(image_idx)
                #Gets the mask to avoid dark areas in segmentation
                mask = get_mask(IOriginal.shape[0:2])
                I = gray2rgb(mask) * IOriginal
                GT = (rgb2gray(dataset.get_ground_truth_data(image_idx).astype(float)) * mask) > 120
                Isegmented = rgb2gray(imread(pathSegmentation + '/' + image + '_our.png').astype(float)) > 0.5


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

                    options = {
                        'weight': 0,  # Weigth of the histogram bins
                        'vdiv': 3,  # one vertical divition
                        'hdiv': 3,  # one horizontal divition
                        'samples': 8,  # number of neighbor samples
                        'mappingtype': 'nri_uniform'  # uniform LBP
                    }

                    Xtmp, Xntmp = Bfx_lbp(I[:, :, 0], Isegmented, options)
                    Xntmp = [name + '_red' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    Xtmp, Xntmp = Bfx_lbp(I[:, :, 1], Isegmented, options)
                    Xntmp = [name + '_green' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    Xtmp, Xntmp = Bfx_lbp(I[:, :, 2], Isegmented, options)
                    Xntmp = [name + '_blue' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    Xtmp, Xntmp = Bfx_lbp(rgb2gray(I), Isegmented, options)
                    Xntmp = [name + '_gray' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    ILabel = label(Isegmented)
                    for region in regionprops(ILabel):
                        minr, minc, maxr, maxc = region.bbox

                    Xtmp = multiblock_lbp(I[:, :, 0], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
                    Xntmp = 'multiblock_lbp_red'

                    Xstack.extend([Xtmp])
                    Xnstack.extend([Xntmp])

                    Xtmp = multiblock_lbp(I[:, :, 1], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
                    Xntmp = 'multiblock_lbp_green'

                    Xstack.extend([Xtmp])
                    Xnstack.extend([Xntmp])

                    Xtmp = multiblock_lbp(I[:, :, 2], minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
                    Xntmp = 'multiblock_lbp_blue'

                    Xstack.extend([Xtmp])
                    Xnstack.extend([Xntmp])

                    Xtmp = multiblock_lbp(rgb2gray(I), minr, minc, int((maxc - minc) / 3), int((maxr - minr) / 3))
                    Xntmp = 'multiblock_lbp_gray'

                    Xstack.extend([Xtmp])
                    Xnstack.extend([Xntmp])

                    X.append(Xstack)
                    if len(Xn) == 0:
                        Xn = Xnstack

                    #print(dataset.get_image_class(image_idx))
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
        #d = (d > 1).astype(int) + 1
        print (Xnclean)
        Xtrain, dtrain, Xtest, dtest = Bds_nostratify(Xclean, d, 0.6)


        op = {
            'm': 10,
            'show': True,
            'b': {'name': 'svm', 'options': {'kernel': 2}}
        }
        s = Bfs_sfs(Xtrain, dtrain, op)
        Xtrain = Xtrain[:, s]
        Xtest = Xtest[:, s]
        Xn = Xnclean[s]

        '''Xtrain = Xtrain[:, [51, 27]]
        Xtest = Xtest[:, [51, 27]]
        #Bio_plotfeatures(Xtrain, dtrain)'''

        b = [
            {'name': 'lda', 'options': {'p': []}},
            {'name': 'maha', 'options': {}},
            {'name': 'qda', 'options': {'p': []}},
            {'name': 'svm', 'options': {'kernel': 1}},
            {'name': 'svm', 'options': {'kernel': 2}},
            {'name': 'knn', 'options': {'k': 5}},
            {'name': 'knn', 'options': {'k': 7}},
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
pathSegmentation = 'results'
magic(imgPath=path,
      imgResults=pathSegmentation,
      method='color',
      segmentationProcess=True,
      featuresProcess=False,
      trainAndTest=False)
