# -*- coding: utf-8 -*-

import numpy as np
from PH2Dataset import PH2Dataset
from matplotlib.image import imread
from skimage.color import rgb2gray
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle, colorbar, savefig
import os
from utils import get_mask

def stat(imgPath, imgResults):

    path = imgPath
    pathSegmentation = os.path.join(imgResults, 'our_segmentation')
    pathResults = os.path.join(imgResults, 'figures')
    TPl = []
    TNl = []
    FNl = []
    FPl = []

    #Set a class to manage the whole dataset
    dataset = PH2Dataset('PH2Dataset')
    #dataset.set_sample(percentage=0.1)
    #dataset.set_sample(image_indices=[0, 50, 2, 5, 198])
    #dataset.set_sample(image_names=['IMD155', 'IMD306', 'IMD382', 'IMD048', 'IMD347', 'IMD386', 'IMD103', 'IMD203', 'IMD312', 'IMD085', 'IMD424', 'IMD384', 'IMD037', 'IMD080', 'IMD369', 'IMD431', 'IMD339', 'IMD031', 'IMD108', 'IMD226'])
    #dataset.exclude_from_sample(image_names=['IMD417'])
    #dataset.exclude_from_sample(image_names=['IMG006', 'IMD008', 'IMD009', 'IMD014', 'IMD019', 'IMD023', 'IMD024', 'IMD032', 'IMD033', 'IMD035', 'IMD037', 'IMD048', 'IMD049', 'IMD058', 'IMD061', 'IMD064', 'IMD085', 'IMD088', 'IMD090', 'IMD091', 'IMD101', 'IMD105', 'IMD112', 'IMD118', 'IMD126', 'IMD135', 'IMD137', 'IMD138', 'IMD147', 'IMD152', 'IMD153', 'IMD154', 'IMD155', 'IMD157', 'IMD159', 'IMD160', 'IMD166', 'IMD168', 'IMD170', 'IMD177', 'IMD182', 'IMD196', 'IMD198', 'IMD200', 'IMD207', 'IMD208', 'IMD219', 'IMD240', 'IMD251', 'IMD254', 'IMD278', 'IMD279', 'IMD280', 'IMD284', 'IMD304', 'IMD339', 'IMD349', 'IMD356', 'IMD360', 'IMD364', 'IMD367', 'IMD368', 'IMD371', 'IMD372', 'IMD375', 'IMD378', 'IMD381', 'IMD382', 'IMD388', 'IMD390', 'IMD397', 'IMD398', 'IMD400', 'IMD403', 'IMD404', 'IMD405', 'IMD406', 'IMD407', 'IMD408', 'IMD409', 'IMD410', 'IMD411', 'IMD413', 'IMD417', 'IMD419', 'IMD420', 'IMD421', 'IMD424', 'IMD425', 'IMD426', 'IMD427', 'IMD430', 'IMD431', 'IMD432', 'IMD433', 'IMD435', 'IMD436'])

    # Mal Segmentadas
    #dataset.set_sample(image_names=['IMG006', 'IMD008', 'IMD009', 'IMD014', 'IMD019', 'IMD023', 'IMD024', 'IMD032', 'IMD033', 'IMD035', 'IMD037', 'IMD048', 'IMD049', 'IMD058', 'IMD061', 'IMD064', 'IMD085', 'IMD088', 'IMD090', 'IMD091', 'IMD101', 'IMD105', 'IMD112', 'IMD118', 'IMD126', 'IMD135', 'IMD137', 'IMD138', 'IMD147', 'IMD152', 'IMD153', 'IMD154', 'IMD155', 'IMD157', 'IMD159', 'IMD160', 'IMD166', 'IMD168', 'IMD170', 'IMD177', 'IMD182', 'IMD196', 'IMD198', 'IMD200', 'IMD207', 'IMD208', 'IMD219', 'IMD240', 'IMD251', 'IMD254', 'IMD278', 'IMD279', 'IMD280', 'IMD284', 'IMD304', 'IMD339', 'IMD349', 'IMD356', 'IMD360', 'IMD364', 'IMD367', 'IMD368', 'IMD371', 'IMD372', 'IMD375', 'IMD378', 'IMD381', 'IMD382', 'IMD388', 'IMD390', 'IMD397', 'IMD398', 'IMD400', 'IMD403', 'IMD404', 'IMD405', 'IMD406', 'IMD407', 'IMD408', 'IMD409', 'IMD410', 'IMD411', 'IMD413', 'IMD417', 'IMD419', 'IMD420', 'IMD421', 'IMD424', 'IMD425', 'IMD426', 'IMD427', 'IMD430', 'IMD431', 'IMD432', 'IMD433', 'IMD435', 'IMD436'])

    # Con pelos
    #dataset.set_sample(image_names=['IMD002', 'IMD003', 'IMD009', 'IMD014', 'IMD024', 'IMD040', 'IMD041', 'IMD048', 'IMD049', 'IMD085', 'IMD101', 'IMD120', 'IMD126', 'IMD146', 'IMD155', 'IMD171', 'IMD177', 'IMD196', 'IMD206', 'IMD251', 'IMD304', 'IMD305', 'IMD306', 'IMD372', 'IMD375', 'IMD405', 'IMD410', 'IMD411'])
    #dataset.set_sample(image_names=['IMD019'])
    #dataset.set_sample(image_names=['IMD035', 'IMD085', 'IMD424', 'IMD105', 'IMD159', 'IMD166'])
    for image_idx in range(dataset.num_images):
        print(image_idx, '/', dataset.num_images)

        image = dataset.image_names[image_idx]
        IOriginal = dataset.get_image_data(image_idx)
        mask = get_mask(IOriginal.shape[0:2])

        GT = (rgb2gray(dataset.get_ground_truth_data(image_idx).astype(float)) * mask) > 120
        Isegmented = rgb2gray(imread(pathSegmentation + '/' + image + '_our.png').astype(float)) > 0.5

        r = np.copy(IOriginal[:, :, :])

        # True Positives -> (mole in both images)
        ITP = np.logical_and(GT, Isegmented)
        TP = np.sum(ITP)

        ii, jj = np.where(ITP)

        r[ii, jj, 0] = 255
        r[ii, jj, 1] = 255
        r[ii, jj, 2] = 255

        #subplot(3,2,4)

        # True Negatives -> (skin in both images)
        ITN = \
            np.logical_and(
                np.logical_not(GT),
                np.logical_not(Isegmented)
            )
        TN = np.sum(ITN)

        ii, jj = np.where(ITN)

        r[ii, jj, 0] = 0
        r[ii, jj, 1] = 0
        r[ii, jj, 2] = 255

        # False Negatives -> (skin in GT mole in our)
        IFN = \
            np.logical_and(
                np.logical_not(GT),
                Isegmented
            )
        FN = np.sum(IFN)

        ii, jj = np.where(IFN)

        r[ii, jj, 0] = 255
        r[ii, jj, 1] = 0
        r[ii, jj, 2] = 0

        # False Positives -> (mole in GT skin in our)
        IFP = \
            np.logical_and(
                GT,
                np.logical_not(Isegmented)
            )
        FP = np.sum(IFP)

        ii, jj = np.where(IFP)

        r[ii, jj, 0] = 0
        r[ii, jj, 1] = 255
        r[ii, jj, 2] = 0


        #imshow(r)
        #show()

        TPl.append(TP)
        TNl.append(TN)
        FNl.append(FN)
        FPl.append(FP)

    TP = sum(TPl)
    TN = sum(TNl)
    FN = sum(FNl)
    FP = sum(FPl)

    sensitivity = TP / (TP + FN)
    specificity = 1 - (TN / (TN + FP))
    accuracy = (TP + TN) / (TP + FN + TN + FP)

    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    print('Accuracy:', accuracy)


path = 'imgs'
pathSegmentation = 'results'
stat(imgPath=path,
      imgResults=pathSegmentation)
