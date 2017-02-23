# -*- coding: utf-8 -*-
import numpy as np
from glob import glob
from abc import ABCMeta, abstractmethod
from os.path import join, splitext, basename, abspath
from balu.DataSelectionAndGeneration import Bds_nostratify


class MissingAttributeException(Exception):
    def __init__(self, attribute):
        super(MissingAttributeException, self).__init__(
            'The <{}> attribute is not contained in the dataset'.format(attribute)
        )


class BaseDataset:
    __metaclass__ = ABCMeta

    def __init__(self, dataset_path):
        self.__dataset_path = abspath(dataset_path)
        self.__image_list = self.load_dataset(self.__dataset_path)
        self.__sample = self.__image_list

    @abstractmethod
    def load_dataset(self, dataset_path):
        '''
        It must return a vector of dictionary in the followed way (keys with * are mandatory):

        image_data = [
                        {
                            'image_filename': <* image filename>,
                            'imageName': <* image name>,
                            'ground_truth_filename': <ground_truth_filename>,
                            'class': <class number the image belong to>,
                            'labels': <vector of string labels associates with the image>
                        },
                        {...}
                    ]


        :param dataset_path: base path for the dataset
        :return: a list of dictionaries containing the dataset information. One entry per image in the dataset
        '''
        return None

    @abstractmethod
    def read_data(self, filename):
        return None

    @abstractmethod
    def has_class(self):
        return False

    @abstractmethod
    def has_labels(self):
        return False

    @abstractmethod
    def has_ground_truth(self):
        return False

    @property
    def image_names(self):
        return [i['imageName'] for i in self.__sample]

    @property
    def num_images(self):
        return len(self.__sample)

    def reset_sample(self):
        self.__sample = self.__image_list

    def set_sample(self, **kwargs):
        if 'percentage' in kwargs:
            p = kwargs['percentage']
            N = len(self.__image_list)
            rn = np.random.rand(N)
            j = np.argsort(rn)[0:int(np.floor(p * N))]
            self.__sample = [self.__image_list[i] for i in j]
        elif 'image_names' in kwargs:
            i_names = kwargs['image_names']
            self.__sample = [data for data in self.__image_list if data['imageName'] in i_names]
        elif 'image_indices' in kwargs:
            image_idxs = kwargs['image_indices']
            self.__sample = [self.__image_list[i] for i in image_idxs]
        else:
            print('No samples was selected. All images will be used.')
            self.reset_sample()

    def exclude_from_sample(self, **kwargs):
        if 'image_names' in kwargs:
            i_names = kwargs['image_names']
            self.__sample = [data for data in self.__sample if data['imageName'] not in i_names]
        elif 'image_indices' in kwargs:
            image_idxs = kwargs['image_indices']
            self.__sample = [self.__sample[i] for i in range(len(self.__sample)) if i not in image_idxs]
        else:
            print('No elements were excluded from the sample.')

    def get_image_data(self, idx_image):
        current_image = self.__sample[idx_image]

        if 'loaded' not in current_image or not current_image['loaded']:
            current_image['data'] = self.read_data(current_image['image_filename'])
            current_image['loaded'] = True

        return current_image['data']

    def get_ground_truth_data(self, idx_image):
        current_image = self.__sample[idx_image]

        if self.has_ground_truth():
            if 'ground_truth_loaded' not in current_image or not current_image['ground_truth_loaded']:
                current_image['ground_truth_data'] = self.read_data(current_image['ground_truth_filename'])
                current_image['ground_truth_loaded'] = True

            return current_image['ground_truth_data']
        else:
            raise MissingAttributeException('ground_truth_data')

    def get_image_class(self, idx_image):
        current_image = self.__sample[idx_image]

        if self.has_class():
            return current_image['class']
        else:
            raise MissingAttributeException('class')

    def get_image_labels(self, idx_image):
        current_image = self.__sample[idx_image]

        if self.has_labels():
            return current_image['labels']
        else:
            raise MissingAttributeException('labels')


class MPEG7Dataset(BaseDataset):
    """
    Concrete implementation of the MPEG-7 Dataset
    """

    def __init__(self, basepath):
        self.image_list = [{
                               'image_filename': fn,
                               'imageName': splitext(basename(fn))[0],
                               'ground_truth_filename': None,
                               'class': None,  # TODO,
                               'labels': None
                           } for fn in glob(join(basepath, 'MPEG7/original/*.gif'))]
