from datasets import BaseDataset
from glob import glob
from os.path import join
from os import listdir
from skimage.io import imread
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


class PH2Dataset(BaseDataset):
    """
    Concrete implementation of the MPEG-7 Dataset
    """

    def __init__(self, base_path):
        BaseDataset.__init__(self, base_path)

    def load_dataset(self, dataset_path):
        image_list = []
        images_base_folder = join(dataset_path, 'PH2 Dataset images')
        image_dirs = [f for f in listdir(images_base_folder) if not f.startswith('.')]

        #Reads the info in the XLSX file
        df = pd.read_excel(join(dataset_path, 'PH2_dataset.xlsx'))
        classes = []
        for i in range(len(image_dirs)):

            cell_common_nevus = df.icol(2).irow(i + 11)
            cell_atypical_nevus = df.icol(3).irow(i + 11)
            cell_melanoma = df.icol(4).irow(i + 11)

            if cell_common_nevus == 'X':
                image_class = 1
            elif cell_atypical_nevus == 'X':
                image_class = 2
            else:
                image_class = 3

            classes.append({
                'name': df.icol(0).irow(i + 11),
                'class': image_class
            })

        for image_folder_name in image_dirs:
            filename = join(images_base_folder, image_folder_name, image_folder_name + '_Dermoscopic_Image',
                            image_folder_name + '.bmp')
            ground_truth_filename = join(images_base_folder, image_folder_name, image_folder_name + '_lesion',
                                         image_folder_name + '_lesion.bmp')

            #image class
            results = list(filter(lambda all_classes: all_classes['name'] == image_folder_name, classes))
            if len(results) > 0:
                c = results[0]['class']
            else:
                c = 1

            image_list.append({
                'image_filename': filename,
                'imageName': image_folder_name,
                'ground_truth_filename': ground_truth_filename,
                'class': c,
                'labels': ['', ''] #TODO: llenar esto
            })

        return image_list

    def read_data(self, filename):
        return imread(filename)

    def has_class(self):
        return True

    def has_labels(self):
        return True

    def has_ground_truth(self):
        return True
