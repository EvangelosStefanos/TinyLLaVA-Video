import os

from PIL import Image, ImageFile
import torch
import ast

from ..utils.data_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoPreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_aspect_ratio = getattr(data_args, 'image_aspect_ratio', None)
        self.image_grid_pinpoints = getattr(data_args, 'image_grid_pinpoints', None)
        self.image_processor = image_processor
    
    def __call__(self, image):
        if image.max() > 1:
            image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.image_processor(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        return image