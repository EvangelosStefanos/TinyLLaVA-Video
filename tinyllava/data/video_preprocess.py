import os

from PIL import Image, ImageFile
import torch
import ast

from ..utils.data_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoPreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_processor = image_processor
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        else:
            if image.max() > 1:
                image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = self.image_processor(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        return image