import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        '''这里需要插值回原size'''
        img_crop = noised_and_cover[0].clone()
        noised_and_cover[0] = nn.functional.interpolate(img_crop, size=(noised_and_cover[1].shape[2],noised_and_cover[1].shape[3]), mode='bilinear')
        return noised_and_cover
