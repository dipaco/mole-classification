# -*- coding: utf-8 -*-
'''
Requeriments:
- Python (We use 3.5).
- Packages: matplotlib, skimage, numpy, mahotas.
'''

import numpy as np
from matplotlib.image import imread
from skimage.color import rgb2gray, label2rgb, gray2rgb
from skimage.segmentation import slic, mark_boundaries
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle, colorbar
from skimage.measure import label, compare_mse
from balu.DataSelectionAndGeneration import Bds_nostratify
import os
import fnmatch
from balu.FeatureExtraction import Bfx_haralick, Bfx_geo, Bfx_basicgeo
from skimage.exposure import histogram
from scipy.special import entr
from scipy.io import savemat, loadmat
from balu.FeatureSelection import Bfs_clean
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
    global dph2
    all_mse = []
    all_jaccard = []

    if featuresProcess:
        X = []
        Xn = []
        d = []
        imagesNames = []

    if segmentationProcess or featuresProcess:
        print("{:10} {:20} {:20}".format('Imagen', 'MSE', 'JACCARD'))
        counter = 0
        for image in fnmatch.filter(os.listdir('imgs'), '*.bmp'):

            if segmentationProcess:
                IOriginal = imread(path + '/' + image)

                # Gets the mask to avoid dark areas in segmentation
                mask = get_mask(IOriginal.shape[0:2])
                I = gray2rgb(mask) * IOriginal
                GT = (rgb2gray(imread(path + 'GT/' + image[:-4] + '_lesion.bmp').astype(float)) * mask) > 120

                #Segment the each mole
                Isegmented = segment(I, mask, method=method)

                auxmse = compare_mse(GT, Isegmented)
                all_mse.append(auxmse)
                auxjacc = compare_jaccard(GT, Isegmented)
                all_jaccard.append(auxjacc)

                print("{:10} {:0.25f} {:0.25f}".format(image[:-4], auxmse, auxjacc))

                if not os.path.exists(pathSegmentation):
                    os.makedirs(pathSegmentation)

                imsave(pathSegmentation + '/' + image[:-4] + '_our.png', Isegmented, cmap='gray')
                counter += 1
                print('{0} / {1}'.format(counter, len(fnmatch.filter(os.listdir('imgs'), '*.bmp'))))

            else: #SEGMENTATION IS DONE AND SAVED
                IOriginal = imread(path + '/' + image)
                #Gets the mask to avoid dark areas in segmentation
                mask = get_mask(IOriginal.shape[0:2])
                I = gray2rgb(mask) * IOriginal
                GT = (rgb2gray(imread(path + 'GT/' + image[:-4] + '_lesion.bmp').astype(float)) * mask) > 120
                Isegmented = rgb2gray(imread(pathSegmentation + '/' + image[:-4] + '_our.png').astype(float)) > 120

            if featuresProcess:

                if (np.sum(Isegmented) > 0):
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

                    J = I[:, :, 0]  # red channel
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented, options)  # Haralick features
                    Xntmp = [name + '_red' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    J = I[:, :, 1]  # green channel
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented, options)  # Haralick features
                    Xntmp = [name + '_green' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    J = I[:, :, 2]  # blue channel
                    Xtmp, Xntmp = Bfx_haralick(J, Isegmented, options)  # Haralick features
                    Xntmp = [name + '_blue' for name in Xntmp]

                    Xstack.extend(Xtmp[0])
                    Xnstack.extend(Xntmp)

                    a, _ = histogram(rgb2gray(I))
                    a = [element / sum(a) for element in a]

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
                    d.extend([dph2[image[:-4]]])
                    imagesNames.extend([image[: -4]])

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
        Xtrain, dtrain, Xtest, dtest = Bds_nostratify(Xclean, d, 0.6)

        b = [
            {'name': 'lda', 'options': {'p': []}},
            {'name': 'maha', 'options': {}},
            {'name': 'qda', 'options': {'p': []}},
            {'name': 'svm', 'options': {'kernel': 1}},
            {'name': 'svm', 'options': {'kernel': 2}},
            {'name': 'knn', 'options': {'k': 5}},
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
    0 - Common Nevus;
    1 - Atypical Nevus;
    2 - Melanoma.
'''
dph2 = {
    'IMD003': 0, 'IMD009': 0, 'IMD016': 0, 'IMD022': 0, 'IMD024': 0, 'IMD025': 0, 'IMD035': 0, 'IMD038': 0, 'IMD042': 0,
    'IMD044': 0, 'IMD045': 0, 'IMD050': 0, 'IMD092': 0, 'IMD101': 0, 'IMD103': 0, 'IMD112': 0, 'IMD118': 0, 'IMD125': 0,
    'IMD132': 0, 'IMD134': 0, 'IMD135': 0, 'IMD144': 0, 'IMD146': 0, 'IMD147': 0, 'IMD150': 0, 'IMD152': 0, 'IMD156': 0,
    'IMD159': 0, 'IMD161': 0, 'IMD162': 0, 'IMD175': 0, 'IMD177': 0, 'IMD182': 0, 'IMD198': 0, 'IMD200': 0, 'IMD010': 0,
    'IMD017': 0, 'IMD020': 0, 'IMD039': 0, 'IMD041': 0, 'IMD105': 0, 'IMD107': 0, 'IMD108': 0, 'IMD133': 0, 'IMD142': 0,
    'IMD143': 0, 'IMD160': 0, 'IMD173': 0, 'IMD176': 0, 'IMD196': 0, 'IMD197': 0, 'IMD199': 0, 'IMD203': 0, 'IMD204': 0,
    'IMD206': 0, 'IMD207': 0, 'IMD208': 0, 'IMD364': 0, 'IMD365': 0, 'IMD367': 0, 'IMD371': 0, 'IMD372': 0, 'IMD374': 0,
    'IMD375': 0, 'IMD378': 0, 'IMD379': 0, 'IMD380': 0, 'IMD381': 0, 'IMD383': 0, 'IMD384': 0, 'IMD385': 0, 'IMD389': 0,
    'IMD390': 0, 'IMD392': 0, 'IMD394': 0, 'IMD395': 0, 'IMD397': 0, 'IMD399': 0, 'IMD400': 0, 'IMD402': 0, 'IMD002': 1,
    'IMD004': 1, 'IMD013': 1, 'IMD015': 1, 'IMD019': 1, 'IMD021': 1, 'IMD027': 1, 'IMD030': 1, 'IMD032': 1, 'IMD033': 1,
    'IMD037': 1, 'IMD040': 1, 'IMD043': 1, 'IMD047': 1, 'IMD048': 1, 'IMD049': 1, 'IMD057': 1, 'IMD075': 1, 'IMD076': 1,
    'IMD078': 1, 'IMD120': 1, 'IMD126': 1, 'IMD137': 1, 'IMD138': 1, 'IMD139': 1, 'IMD140': 1, 'IMD149': 1, 'IMD153': 1,
    'IMD157': 1, 'IMD164': 1, 'IMD166': 1, 'IMD169': 1, 'IMD171': 1, 'IMD210': 1, 'IMD347': 1, 'IMD155': 1, 'IMD376': 1,
    'IMD006': 1, 'IMD008': 1, 'IMD014': 1, 'IMD018': 1, 'IMD023': 1, 'IMD031': 1, 'IMD036': 1, 'IMD154': 1, 'IMD170': 1,
    'IMD226': 1, 'IMD243': 1, 'IMD251': 1, 'IMD254': 1, 'IMD256': 1, 'IMD278': 1, 'IMD279': 1, 'IMD280': 1, 'IMD304': 1,
    'IMD305': 1, 'IMD306': 1, 'IMD312': 1, 'IMD328': 1, 'IMD331': 1, 'IMD339': 1, 'IMD356': 1, 'IMD360': 1, 'IMD368': 1,
    'IMD369': 1, 'IMD370': 1, 'IMD382': 1, 'IMD386': 1, 'IMD388': 1, 'IMD393': 1, 'IMD396': 1, 'IMD398': 1, 'IMD427': 1,
    'IMD430': 1, 'IMD431': 1, 'IMD432': 1, 'IMD433': 1, 'IMD434': 1, 'IMD436': 1, 'IMD437': 1, 'IMD058': 2, 'IMD061': 2,
    'IMD063': 2, 'IMD064': 2, 'IMD065': 2, 'IMD080': 2, 'IMD085': 2, 'IMD088': 2, 'IMD090': 2, 'IMD091': 2, 'IMD168': 2,
    'IMD211': 2, 'IMD219': 2, 'IMD240': 2, 'IMD242': 2, 'IMD284': 2, 'IMD285': 2, 'IMD348': 2, 'IMD349': 2, 'IMD403': 2,
    'IMD404': 2, 'IMD405': 2, 'IMD407': 2, 'IMD408': 2, 'IMD409': 2, 'IMD410': 2, 'IMD413': 2, 'IMD417': 2, 'IMD418': 2,
    'IMD419': 2, 'IMD406': 2, 'IMD411': 2, 'IMD420': 2, 'IMD421': 2, 'IMD423': 2, 'IMD424': 2, 'IMD425': 2, 'IMD426': 2,
    'IMD429': 2, 'IMD435': 2
}

path = 'imgs'
pathSegmentation = 'our_segmentation'
magic(imgPath=path,
      imgSegPath=pathSegmentation,
      method='haralick',
      segmentationProcess=True,
      featuresProcess=True,
      trainAndTest=True)