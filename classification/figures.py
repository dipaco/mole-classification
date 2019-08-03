import matplotlib
matplotlib.use('TkAgg')
from datasets.PH2Dataset import PH2Dataset
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import os

# creation of the dataset
dataset = PH2Dataset('/Users/dipaco/Documents/Repos/mole-classification/segmentation/PH2Dataset')

# Path to the segmented images
segmentation_folder = '/Users/dipaco/Documents/Repos/mole-classification/segmentation/results_final/our_segmentation'
figures_folder = '/Users/dipaco/Trash/Figuras_skin'

for i in range(dataset.num_images):

    # Reads the image and its segmentation
    image = dataset.get_image_data(i)
    mascara = plt.imread(os.path.join(segmentation_folder, dataset.image_names[i] + '_our.png'))[..., 0] > 0
    mascara = mascara.astype(int)
    #dataset.get_ground_truth_data(i)

    # Find contours
    contours = measure.find_contours(mascara, 0.8)

    # Calc all the region properties
    props = measure.regionprops(mascara)

    # Set the centroid and orientation
    y0, x0 = props[0].centroid
    orientation = -props[0].orientation # the negative orientation form the angle with the x axis going down (what we need)

    normals = [
        {'axis': 'major_axis', 'n': np.array([[np.sin(orientation), -np.cos(orientation), 0]])},
        {'axis': 'minor_axis', 'n': np.array([[np.cos(orientation), np.sin(orientation), 0]])},
    ]

    for e in normals:
        fig, ax = plt.subplots()

        # Plot the image in the background to visualize the results
        ax.imshow(image, cmap=plt.cm.gray)

        n = e['n']
        print(dataset.image_names[i], 'class: ', dataset.get_image_class(i), 'image: ', i)

        contour = contours[0]

        # Plot the contour
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        # Plot both major and minor axis
        x1 = x0 + np.cos(orientation) * 0.5 * props[0].major_axis_length
        y1 = y0 + np.sin(orientation) * 0.5 * props[0].major_axis_length
        x2 = x0 - np.sin(orientation) * 0.5 * props[0].minor_axis_length
        y2 = y0 + np.cos(orientation) * 0.5 * props[0].minor_axis_length

        if e['axis'] == 'major_axis':
            ax.plot((x0, x1), (y0, y1), 'black', linewidth=2)
        else:
            ax.plot((x0, x2), (y0, y2), 'black', linewidth=2)

        ax.plot(x0, y0, '.g', markersize=7)

        # -> Translation
        T = np.eye(3)
        T[0:2, 2] = [-x0, -y0]
        # -> Reflection
        R = np.eye(3) - 2 * n.T @ n

        # apply the transformation to the every point in the contour
        contour_r = np.stack((contour[:, 1], contour[:, 0], np.ones(contour[:, 0].size)))
        contour_r = (np.linalg.inv(T) @ (R @ (T @ contour_r))).T # applys the

        # Plots the reflected contour around the line defined by the vector n (normal to the line)
        ax.plot(contour_r[:, 0], contour_r[:, 1], linewidth=2, color='r')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(os.path.join(figures_folder, 'class_{}_{}_{}.png'.format(dataset.get_image_class(i), dataset.image_names[i], e['axis'])))
        plt.close()
        #plt.show()