from datasets import BaseDataset
from glob import glob
from os.path import join, splitext
from os import listdir
from skimage.io import imread
from skimage.transform import rescale
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


class ISICDataset(BaseDataset):
    """
    Concrete implementation of the MPEG-7 Dataset
    """

    def __init__(self, base_path):
        BaseDataset.__init__(self, base_path)

    def load_dataset(self, dataset_path):
        image_list = []
        images_base_folder = join(dataset_path, 'ISIC-2017_Training_Data')
        images_gt_base_folder = join(dataset_path, 'ISIC-2017_Training_Part1_GroundTruth')
        all_images = [f for f in listdir(images_base_folder) if not f.startswith('.') and 'superpixels' not in f and not f.endswith('.csv')]

        #Reads the info in the XLSX file
        #df = pd.read_excel(join(dataset_path, 'PH2_dataset.xlsx'))

        for f in all_images:

            image_base_name, ext = splitext(f)

            filename = join(images_base_folder, f)
            ground_truth_filename = join(images_gt_base_folder, image_base_name + '_segmentation.png')

            image_list.append({
                'image_filename': filename,
                'imageName': image_base_name,
                'ground_truth_filename': ground_truth_filename,
                'class': 1,
                'labels': ['Melanoma', ''] #TODO: llenar esto
            })

        return image_list

    def read_data(self, filename):
        return rescale(imread(filename), 0.5)

    def has_class(self):
        return True

    def has_labels(self):
        return True

    def has_ground_truth(self):
        return True
