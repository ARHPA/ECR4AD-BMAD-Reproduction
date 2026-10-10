import torch
import scipy.stats
import numpy 
from dataloader.samplers import CategoriesSampler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score, precision_recall_curve, auc, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader
from skimage import measure
import torch.nn as nn
import json
from tqdm import tqdm
import wandb
import statistics
from utils import get_gaussian_kernel
from torch.nn import functional as F
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def tpr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def fpr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def find_percentile_threshold(scores, percentile=95):
    scores =  numpy.array(scores)
    threshold = numpy.percentile(scores, percentile)

    return threshold

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def compute_pro(masks: numpy.ndarray, amaps: numpy.ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """ 
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = numpy.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    
    for th in numpy.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1
        
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = numpy.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        #df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{"pro": statistics.mean(pros), "fpr": fpr, "threshold": th}])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

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

def optimal_roc_threshold(y_true, y_probabilities):
    """
    Finds the optimal threshold for ROC curve.

    Args:
        y_true (array-like): True labels.
        y_probabilities (array-like): Predicted probabilities.

    Returns:
        float: Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probabilities)
    # Calculate Youden's J statistic
    j_scores = tpr - fpr
    # Find the index of the maximum J statistic
    max_j_index = numpy.argmax(j_scores)
    # Return the corresponding threshold
    optimal_threshold = thresholds[max_j_index]

    return optimal_threshold

def patch_evaluation(model, data_loader, max_ratio = 0, resize_mask = None, threshold = None, find_threshold = False):
    model.eval()
    gt_list = []
    pr_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size = 5, sigma = 4).cuda()

    file_name_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            img = batch[0].cuda()
            label = batch[1]

            file_name_list = file_name_list + list(batch[2])
            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            gt_list.append(label)

            if max_ratio == 0:
                score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                score = score.mean(dim=1)
            pr_list.append(score)

        gt_list = torch.cat(gt_list).flatten().cpu().numpy()
        pr_list = torch.cat(pr_list).flatten().cpu().numpy()

        if find_threshold:
            return find_percentile_threshold(pr_list)

        if threshold is None:
            threshold = optimal_roc_threshold(gt_list, pr_list)

        if numpy.unique(gt_list).shape[0] > 1:
            auroc = roc_auc_score(gt_list, pr_list)
            ap = average_precision_score(gt_list, pr_list)
            f1 = f1_score_max(gt_list, pr_list)

            pred_binary = (pr_list >= threshold).astype(int)
            fpr = fpr_score(gt_list, pred_binary)
            tpr = tpr_score(gt_list, pred_binary)

            return (auroc, ap, f1, fpr, tpr)
        
        else:
            pred_binary = (pr_list >= threshold).astype(int)
            fpr = fpr_score(gt_list, pred_binary)
            #tpr = tpr_score(gt_list, pred_binary)

            return (fpr,)
