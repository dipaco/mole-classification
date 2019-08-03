# -*- coding: utf-8 -*-
'''
Requeriments:
- Python (We use 3.5).
- Packages: matplotlib, skimage, numpy, mahotas.
'''

import numpy as np
from PH2Dataset import PH2Dataset
from ISICDataset import ISICDataset
from skimage.color import rgb2gray, gray2rgb
from matplotlib.pyplot import imshow, figure, title, imsave, savefig, subplot
from skimage.measure import compare_mse
import os

#Segmentation
from segmentation import segment
from utils import get_mask, compare_jaccard


def main(dataset, results_path, method='color', k=400):

    segmentation_path = os.path.join(results_path, 'our_segmentation')
    figures_path = os.path.join(results_path, 'figures')
    all_mse = []
    all_jaccard = []
    all_acc = []

    print("{:10} {:20} {:20}".format('Imagen', 'MSE', 'JACCARD'))
    for image_idx in range(dataset.num_images):

        image = dataset.image_names[image_idx]

        # reads the image information from the dataset
        original_image = dataset.get_image_data(image_idx)

        # Gets the mask to avoid dark areas in segmentation
        mask = get_mask(original_image.shape[0:2])
        I = gray2rgb(mask) * original_image
        GT = (rgb2gray(dataset.get_ground_truth_data(image_idx).astype(float)*255) * mask) > 0

        #Segment the each mole
        print('Segmenting image {0} ({1} / {2})'.format(dataset.image_names[image_idx], image_idx + 1, dataset.num_images))
        Isegmented, LMerged, Islic2, IOtsu, Superpixels = segment(I, mask, method=method, k=k)

        auxmse = compare_mse(GT, Isegmented)
        all_mse.append(auxmse)
        aux_jaccard = compare_jaccard(GT, Isegmented)
        aux_acc = 1.0 - np.sum(np.logical_xor(GT, Isegmented)) / float(GT.size)
        all_jaccard.append(aux_jaccard)
        all_acc.append(aux_acc)

        print("Image name, MSE, JACCARD_IDX, ACC")
        print("{:10} {:0.25f} {:0.25f} {:0.25f}".format(image, auxmse, aux_jaccard, aux_acc))

        if not os.path.exists(segmentation_path):
            os.makedirs(segmentation_path)
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        subplot(2, 3, 1)
        title('Original + Superpixels')
        imshow(Superpixels)
        subplot(2, 3, 2)
        title('Ground Truth')
        imshow(GT, cmap='gray')
        subplot(2, 3, 3)
        title('Our Segmentation')
        imshow(Isegmented, cmap='gray')
        subplot(2, 3, 4)
        title('Labels')
        imshow(LMerged)
        subplot(2, 3, 5)
        title('Merged Superpixels')
        imshow(Islic2)
        subplot(2, 3, 6)
        title('Otsu')
        imshow(IOtsu, cmap='gray')
        savefig(figures_path + '/' + image + '_all.png')

        imsave(segmentation_path + '/' + image + '_our.png', 255*Isegmented.astype(int), cmap='gray')

        C = np.zeros_like(Isegmented).astype(int)
        a = np.where(np.logical_and(GT, Isegmented)) # TP
        b = np.where(np.logical_and(GT, np.logical_not(Isegmented))) #FN
        d = np.where(np.logical_and(Isegmented, np.logical_not(GT))) #FP
        C[a] = 1
        C[b] = 2
        C[d] = 3

        figure()
        title('Seg. comparison')
        imshow(C)
        savefig(figures_path + '/' + image + '_k_{}_seg_comp.png'.format(k))

        figure()
        title('SLIC Segmentation, k = {}'.format(k))
        imshow(Superpixels)
        savefig(figures_path + '/' + image + '_k_{}_seg.png'.format(k))

        figure()
        title('Merged superpixels')
        imshow(Islic2)
        savefig(figures_path + '/' + image + '_k_{}_merged.png'.format(k))

        figure()
        title('Otsu')
        imshow(IOtsu, cmap='gray')
        savefig(figures_path + '/' + image + '_k_{}_otsu.png'.format(k))

    print('jaccard overall: {}'.format(np.mean(np.array(all_jaccard))))
    print('acc. overall: {}'.format(np.mean(np.array(all_acc))))


'''
Clinical Diagnosis:
    1 - Common Nevus;
    2 - Atypical Nevus;
    3 - Melanoma.
'''

path = 'imgs'
path_to_segmentation = './results'
ph2_dataset_path = './PH2Dataset'
isic_dataset = '/Users/dipaco/Documents/Datasets/ISIC-2017'

#Set a class to manage the whole dataset
#dataset = PH2Dataset(ph2_dataset_path)
dataset = ISICDataset(isic_dataset)

min_ = 900
max_ = 1000
dataset.set_sample(image_indices=range(min_, max_))

# Uncomment the following line to use just images: 'IMD242', 'IMD368', 'IMD306', instead of the whole dataset
#dataset.set_sample(image_names=['IMD242', 'IMD368', 'IMD306'])

print(min_, max_)
main(dataset=dataset, results_path=path_to_segmentation, method='color', k=400)
print(min_, max_)