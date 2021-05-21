from __future__ import division
from PIL import Image, ImageDraw
from torch.autograd import Variable
from util import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def letterbox_image(img, input_dim):
    # resize image with unchanged aspect ration using padding
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = input_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((input_dim[1], input_dim[0], 3), 128)
    canvas[
        (h - new_h) // 2 : (h - new_h) // 2 + new_h,
        (w - new_w) // 2 : (w - new_w) // 2 + new_w,
        :,
    ] = resized_img
    return canvas


def prep_image(img, input_dim):
    # Prepare image to be feed to model.
    orig_img = cv2.imread(img)
    dim = orig_img.shape[1], orig_img.shape[0]
    img = letterbox_image(orig_img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_img, dim
