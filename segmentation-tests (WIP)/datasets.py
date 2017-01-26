# -*- coding: utf-8 -*-
import numpy as np
from glob import glob
from abc import ABCMeta, abstractmethod
from skimage.io import imread
from os.path import join, splitext, basename


class BaseDataset(metaclass=ABCMeta):
    @property
    def imageNames(self):
        return [i['imageName'] for i in self.image_list]

    @property
    def numImages(self):
        return len(self.imageNames)

    def getImageData(self, idxImage):
        current_image = self.image_list[idxImage]
        if 'loaded' not in current_image:
            current_image['data'] = imread(current_image['filename'])
            current_image['loaded'] = True

        return current_image['data']

class PH2Dataset(BaseDataset):
    """
    Concrete implementation of the MPEG-7 Dataset
    """

    def __init__(self, basepath):
        self.image_list = [{
                                'filename': fn,
                                'imageName': splitext(basename(fn))[0]
                           } for fn in glob(join(basepath, 'MPEG7/original/*.gif'))]

