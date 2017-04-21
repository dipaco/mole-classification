# -*- coding: utf-8 -*-

import numpy as np
from PH2Dataset import PH2Dataset
from matplotlib.image import imread
from skimage.color import rgb2gray
from matplotlib.pyplot import show, imshow, subplot, figure, title, imsave, suptitle, colorbar, savefig
import os
from utils import get_mask


def get_sensitivity(TP, FN):
    return TP / (TP + FN)


def get_specificity(TN, FP):
    return 1 - (TN / (TN + FP))


def get_specificity_wiki(TN, FP):
    return TN / (TN + FP)


def get_accuracy(TP, TN, FN, FP):
    return (TP + TN) / (TP + FN + TN + FP)


def allM(TP, TN, FN, FP):
    sensitivity = get_sensitivity(TP, FN)
    specificity = get_specificity(TN, FP)
    accuracy = get_accuracy(TP, TN, FN, FP)
    specificity_wiki = get_specificity_wiki(TN, FP)
    # TODO: F-measure

    return sensitivity, specificity, accuracy, specificity_wiki


def stat(imgPath, imgResults):

    path = imgPath
    pathSegmentation = os.path.join(imgResults, 'our_segmentation')
    pathResults = os.path.join(imgResults, 'figures')
    sensitivityAll = np.array([])
    specificityAll = np.array([])
    specificity_wikiAll = np.array([])
    accuracyAll = np.array([])

    #Set a class to manage the whole dataset
    dataset = PH2Dataset('PH2Dataset')
    #dataset.set_sample(percentage=0.1)
    #dataset.set_sample(image_indices=[0, 50, 2, 5, 198])
    #dataset.set_sample(image_names=['IMD155', 'IMD306', 'IMD382', 'IMD048', 'IMD347', 'IMD386', 'IMD103', 'IMD203', 'IMD312', 'IMD085', 'IMD424', 'IMD384', 'IMD037', 'IMD080', 'IMD369', 'IMD431', 'IMD339', 'IMD031', 'IMD108', 'IMD226'])
    #dataset.exclude_from_sample(image_names=['IMD417'])
    #dataset.exclude_from_sample(image_names=['IMG006', 'IMD008', 'IMD009', 'IMD014', 'IMD019', 'IMD023', 'IMD024', 'IMD032', 'IMD033', 'IMD035', 'IMD037', 'IMD048', 'IMD049', 'IMD058', 'IMD061', 'IMD064', 'IMD085', 'IMD088', 'IMD090', 'IMD091', 'IMD101', 'IMD105', 'IMD112', 'IMD118', 'IMD126', 'IMD135', 'IMD137', 'IMD138', 'IMD147', 'IMD152', 'IMD153', 'IMD154', 'IMD155', 'IMD157', 'IMD159', 'IMD160', 'IMD166', 'IMD168', 'IMD170', 'IMD177', 'IMD182', 'IMD196', 'IMD198', 'IMD200', 'IMD207', 'IMD208', 'IMD219', 'IMD240', 'IMD251', 'IMD254', 'IMD278', 'IMD279', 'IMD280', 'IMD284', 'IMD304', 'IMD339', 'IMD349', 'IMD356', 'IMD360', 'IMD364', 'IMD367', 'IMD368', 'IMD371', 'IMD372', 'IMD375', 'IMD378', 'IMD381', 'IMD382', 'IMD388', 'IMD390', 'IMD397', 'IMD398', 'IMD400', 'IMD403', 'IMD404', 'IMD405', 'IMD406', 'IMD407', 'IMD408', 'IMD409', 'IMD410', 'IMD411', 'IMD413', 'IMD417', 'IMD419', 'IMD420', 'IMD421', 'IMD424', 'IMD425', 'IMD426', 'IMD427', 'IMD430', 'IMD431', 'IMD432', 'IMD433', 'IMD435', 'IMD436'])

    # Sanos
    '''
    dataset.set_sample(
        image_names=['IMD003', 'IMD009', 'IMD016', 'IMD022', 'IMD024', 'IMD025', 'IMD035', 'IMD038', 'IMD042', 'IMD044',
                     'IMD045', 'IMD050', 'IMD092', 'IMD101', 'IMD103', 'IMD112', 'IMD118', 'IMD125', 'IMD132', 'IMD134',
                     'IMD135', 'IMD144', 'IMD146', 'IMD147', 'IMD150', 'IMD152', 'IMD156', 'IMD159', 'IMD161', 'IMD162',
                     'IMD175', 'IMD177', 'IMD182', 'IMD198', 'IMD200', 'IMD010', 'IMD017', 'IMD020', 'IMD039', 'IMD041',
                     'IMD105', 'IMD107', 'IMD108', 'IMD133', 'IMD142', 'IMD143', 'IMD160', 'IMD173', 'IMD176', 'IMD196',
                     'IMD197', 'IMD199', 'IMD203', 'IMD204', 'IMD206', 'IMD207', 'IMD208', 'IMD364', 'IMD365', 'IMD367',
                     'IMD371', 'IMD372', 'IMD374', 'IMD375', 'IMD378', 'IMD379', 'IMD380', 'IMD381', 'IMD383', 'IMD384',
                     'IMD385', 'IMD389', 'IMD390', 'IMD392', 'IMD394', 'IMD395', 'IMD397', 'IMD399', 'IMD400',
                     'IMD402'])
    '''
    # Atipicos
    '''
    dataset.set_sample(
        image_names=['IMD002', 'IMD004', 'IMD013', 'IMD015', 'IMD019', 'IMD021', 'IMD027', 'IMD030', 'IMD032', 'IMD033',
                     'IMD037', 'IMD040', 'IMD043', 'IMD047', 'IMD048', 'IMD049', 'IMD057', 'IMD075', 'IMD076', 'IMD078',
                     'IMD120', 'IMD126', 'IMD137', 'IMD138', 'IMD139', 'IMD140', 'IMD149', 'IMD153', 'IMD157', 'IMD164',
                     'IMD166', 'IMD169', 'IMD171', 'IMD210', 'IMD347', 'IMD155', 'IMD376', 'IMD006', 'IMD008', 'IMD014',
                     'IMD018', 'IMD023', 'IMD031', 'IMD036', 'IMD154', 'IMD170', 'IMD226', 'IMD243', 'IMD251', 'IMD254',
                     'IMD256', 'IMD278', 'IMD279', 'IMD280', 'IMD304', 'IMD305', 'IMD306', 'IMD312', 'IMD328', 'IMD331',
                     'IMD339', 'IMD356', 'IMD360', 'IMD368', 'IMD369', 'IMD370', 'IMD382', 'IMD386', 'IMD388', 'IMD393',
                     'IMD396', 'IMD398', 'IMD427', 'IMD430', 'IMD431', 'IMD432', 'IMD433', 'IMD434', 'IMD436',
                     'IMD437'])
    '''
    # Melanoma
    '''
    dataset.set_sample(
        image_names=['IMD058', 'IMD061', 'IMD063', 'IMD064', 'IMD065', 'IMD080', 'IMD085', 'IMD088', 'IMD090', 'IMD091',
                     'IMD168', 'IMD211', 'IMD219', 'IMD240', 'IMD242', 'IMD284', 'IMD285', 'IMD348', 'IMD349', 'IMD403',
                     'IMD404', 'IMD405', 'IMD407', 'IMD408', 'IMD409', 'IMD410', 'IMD413', 'IMD417', 'IMD418', 'IMD419',
                     'IMD406', 'IMD411', 'IMD420', 'IMD421', 'IMD423', 'IMD424', 'IMD425', 'IMD426', 'IMD429',
                     'IMD435'])
    '''
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

        sensitivity, specificity, accuracy, specificity_wiki = allM(TP, TN, FN, FP)

        sensitivityAll = np.append(sensitivityAll, sensitivity)
        specificity_wikiAll = np.append(specificity_wikiAll, specificity_wiki)
        specificityAll = np.append(specificityAll, specificity)
        accuracyAll = np.append(accuracyAll, accuracy)

        #imshow(r)
        #show()

    sensitivity = np.mean(sensitivityAll)
    sensitivityStd = np.std(sensitivityAll)

    specificity = np.mean(specificityAll)
    specificityStd = np.std(specificityAll)

    specificity_wiki = np.mean(specificity_wikiAll)
    specificity_wikiStd = np.std(specificity_wikiAll)

    accuracy = np.mean(accuracyAll)
    accuracyStd = np.std(accuracyAll)

    print('Sensitivity:', sensitivity, '±', sensitivityStd)
    print('Specificity:', specificity, '±', specificityStd)
    print('Specificity_wiki:', specificity_wiki, '±', specificity_wikiStd)
    print('Accuracy:', accuracy, '±', accuracyStd)


path = 'imgs'
pathSegmentation = 'results'
stat(imgPath=path,
      imgResults=pathSegmentation)
