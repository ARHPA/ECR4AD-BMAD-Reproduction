import os
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math
import random
import cv2
import numpy as np
import pandas as pd
from skimage import measure, io
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import statistics

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def compute_sample_pro_fpr(binary_amap: np.ndarray, mask: np.ndarray) -> float:
    """Compute the PRO for a single binary_amap and mask pair."""
    pros = []
    for region in measure.regionprops(measure.label(mask)):
        axes0_ids = region.coords[:, 0]
        axes1_ids = region.coords[:, 1]
        tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
        pros.append(tp_pixels / region.area)
    return pros

def seed_everything(seed = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_autocast_enabled(False)

def get_gaussian_kernel(kernel_size = 3, sigma = 2, channels = 1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
            groups=channels, bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):
    
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

def hsv_filter(image, coverage = 45):
    """
    Apply HSV filtering to an image with specified ranges.
    Hue range: [90, 180]
    Saturation range: [8, 255]
    Value range: [103, 255]
    Minimum tissue coverage: 45%
    
    Args:
        image: Input image in either numpy array (BGR format) or PIL.Image format
        
    Returns:
        Binary mask where 1 indicates pixels within the specified HSV ranges
        Returns all zeros if tissue coverage is less than 45%
    """
    # Handle PIL Image
    if isinstance(image, Image.Image):
        # Convert PIL Image to numpy array and BGR format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges
    lower_bound = np.array([90, 8, 103])
    upper_bound = np.array([180, 255, 255])
    
    # Create mask for pixels within the specified ranges
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Convert to binary mask (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Calculate tissue coverage percentage
    coverage = np.sum(binary_mask) / binary_mask.size * 100
    # If coverage is less than 45%, return False
    if coverage < 45:
        return False
    
    return True

def mask_filter(mask_path, original_width, original_height):
    Image.MAX_IMAGE_PIXELS = None
    raw_mask = io.imread(mask_path)
    gray_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
    binary_mask = (gray_mask == 255).astype(np.float32)
    binary_mask = cv2.resize(binary_mask, (slide_w, slide_h), interpolation=cv2.INTER_NEAREST)

    binary_mask[binary_mask >= 0.5] = 1
    binary_mask[binary_mask < 0.5] = 0
