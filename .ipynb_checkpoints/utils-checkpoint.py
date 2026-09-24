import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math
from scipy.ndimage import gaussian_filter
import pickle

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    loss = loss / len(a)
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
            point_dist = torch.nan_to_num(point_dist, nan=0.0, posinf=0.0, neginf=0.0)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    # loss = loss / len(a)
    return loss


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    if not math.isfinite(float(p)):
        p = 0.9
    p = min(max(float(p), 0.0), 0.999999)
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        k = max(1, int(point_dist.numel() * (1 - p)))
        thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def regional_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += point_dist.mean()

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def global_cosine_focal(a, b, p=0.9, alpha=2., min_grad=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()

        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh

        focal_factor = focal_factor ** alpha
        focal_factor = torch.clip(focal_factor, min=min_grad)

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad_v2, factor=focal_factor)
        b_.register_hook(partial_func)

    return loss


def regional_cosine_focal(a, b, p=0.9, alpha=2.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh
        focal_factor = focal_factor ** alpha

        loss += (point_dist * focal_factor.detach()).mean()

    return loss


def regional_cosine_hm(a, b, p=0.9):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        L = point_dist[point_dist >= thresh]
        loss += L.mean()

    return loss


def region_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * (a_map - norm_factor[0][i]) / (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


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


def build_reference_bank(model, dataloader, device):
    model.eval()
    ref_bank = None
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            en, _ = model(img)
            if ref_bank is None:
                ref_bank = [[] for _ in en]
            for i, feat in enumerate(en):
                feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
                ref_bank[i].append(F.normalize(feat_flat, dim=1).cpu())

    if ref_bank is None:
        return None
    return [torch.cat(layer_bank, dim=0).to(device) for layer_bank in ref_bank]


def cal_reference_map(fs_list, ref_bank, out_size=224, chunk_size=1024):
    if ref_bank is None:
        return None
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    ref_maps = []
    for feat, bank in zip(fs_list, ref_bank):
        b, c, h, w = feat.shape
        query = feat.permute(0, 2, 3, 1).reshape(-1, c)
        query = F.normalize(query, dim=1)
        min_dist_chunks = []
        for chunk in torch.split(query, chunk_size, dim=0):
            sim = torch.matmul(chunk, bank.t())
            min_dist_chunks.append(1.0 - sim.max(dim=1)[0])
        ref_map = torch.cat(min_dist_chunks, dim=0).view(b, 1, h, w)
        ref_map = F.interpolate(ref_map, size=out_size, mode='bilinear', align_corners=False)
        ref_maps.append(ref_map)

    ref_map = torch.cat(ref_maps, dim=1).mean(dim=1, keepdim=True)
    ref_flat = ref_map.flatten(1)
    ref_min = ref_flat.min(dim=1)[0].view(-1, 1, 1, 1)
    ref_max = ref_flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (ref_map - ref_min) / (ref_max - ref_min + torch.finfo(ref_map.dtype).eps)


def map_normalization(fs_list, ft_list, start=0.5, end=0.95):
    start_list = []
    end_list = []
    with torch.no_grad():
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            start_list.append(torch.quantile(a_map, q=start).item())
            end_list.append(torch.quantile(a_map, q=end).item())

    return [start_list, end_list]


def cal_anomaly_map_v2(fs_list, ft_list, out_size=224, amap_mode='add'):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size // 4, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)

    anomaly_map = torch.stack(a_map_list, dim=-1).sum(-1)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=False)
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()

    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True, norm_factor=None, feature_used='all',
               max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            en, de = model(img)

            if feature_used == 'trained':
                anomaly_map, _ = cal_anomaly_map(en[3:], de[3:], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            elif feature_used == 'freezed':
                anomaly_map, _ = cal_anomaly_map(en[:3], de[:3], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            else:
                anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            if max_ratio <= 0:
                sp_score = anomaly_map.max()
            else:
                anomaly_map = anomaly_map.ravel()
                sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
                sp_score = sp_score.mean()
            pr_list_sp.append(sp_score)
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def _image_level_score_candidates(anomaly_map, max_ratio):
    anomaly_flat = anomaly_map.flatten(1)
    num_pixels = anomaly_flat.shape[1]
    sorted_score = torch.sort(anomaly_flat, dim=1, descending=True)[0]

    def topk_mean(ratio):
        k = max(1, int(num_pixels * ratio))
        return sorted_score[:, :k].mean(dim=1)

    if max_ratio == 0:
        base_score = sorted_score[:, 0]
    else:
        base_score = topk_mean(max_ratio)

    top_001 = topk_mean(0.001)
    top_0025 = topk_mean(0.0025)
    top_005 = topk_mean(0.005)
    top_01 = topk_mean(0.01)
    top_02 = topk_mean(0.02)
    top_05 = topk_mean(0.05)
    peak_score = sorted_score[:, 0]
    std_score = anomaly_flat.std(dim=1)
    mean_score = anomaly_flat.mean(dim=1)
    eps = torch.finfo(anomaly_flat.dtype).eps
    q995_score = torch.quantile(anomaly_flat, q=0.995, dim=1)
    q99_score = torch.quantile(anomaly_flat, q=0.99, dim=1)
    q98_score = torch.quantile(anomaly_flat, q=0.98, dim=1)

    return [
        base_score,
        peak_score,
        top_001,
        top_0025,
        top_005,
        top_01,
        top_02,
        top_05,
        q995_score,
        q99_score,
        q98_score,
        (peak_score - mean_score) / (std_score + eps),
        (top_001 - mean_score) / (std_score + eps),
        (top_0025 - mean_score) / (std_score + eps),
        (top_005 - mean_score) / (std_score + eps),
        (top_01 - mean_score) / (std_score + eps),
        0.75 * base_score + 0.25 * peak_score,
        0.50 * top_001 + 0.50 * top_01,
        0.60 * top_0025 + 0.40 * top_02,
        0.70 * top_005 + 0.30 * top_02,
        0.50 * top_01 + 0.50 * top_05,
        q995_score + 0.10 * std_score,
        q99_score + 0.10 * std_score,
        base_score + 0.10 * std_score,
        base_score + 0.05 * mean_score,
    ]


def _foreground_score_candidates(anomaly_map, en_features, max_ratio):
    activation = torch.norm(en_features, p=2, dim=1, keepdim=True)
    activation = F.interpolate(activation, size=anomaly_map.shape[-2:], mode='bilinear', align_corners=False)
    activation_flat = activation.flatten(1)
    act_min = activation_flat.min(dim=1)[0].view(-1, 1, 1, 1)
    act_max = activation_flat.max(dim=1)[0].view(-1, 1, 1, 1)
    fg_weight = (activation - act_min) / (act_max - act_min + torch.finfo(activation.dtype).eps)

    soft_fg_map = anomaly_map * (0.15 + 0.85 * fg_weight)
    fg_flat = fg_weight.flatten(1)
    hard_fg_mask = (fg_weight > torch.quantile(fg_flat, q=0.35, dim=1).view(-1, 1, 1, 1)).float()
    strict_fg_mask = (fg_weight > torch.quantile(fg_flat, q=0.50, dim=1).view(-1, 1, 1, 1)).float()
    core_fg_mask = (fg_weight > torch.quantile(fg_flat, q=0.65, dim=1).view(-1, 1, 1, 1)).float()
    hard_fg_map = anomaly_map * hard_fg_mask
    strict_fg_map = anomaly_map * strict_fg_mask
    core_fg_map = anomaly_map * core_fg_mask

    candidates = []
    candidates.extend(_image_level_score_candidates(soft_fg_map, max_ratio))
    candidates.extend(_image_level_score_candidates(hard_fg_map, max_ratio))
    candidates.extend(_image_level_score_candidates(strict_fg_map, max_ratio))
    candidates.extend(_image_level_score_candidates(core_fg_map, max_ratio))

    anomaly_flat = anomaly_map.flatten(1)
    num_pixels = anomaly_flat.shape[1]
    k_small = max(1, int(num_pixels * 0.0025))
    k_mid = max(1, int(num_pixels * 0.01))
    k_large = max(1, int(num_pixels * 0.02))
    fg_top = torch.sort((anomaly_flat * fg_flat), dim=1, descending=True)[0]
    bg_top = torch.sort((anomaly_flat * (1.0 - fg_flat)), dim=1, descending=True)[0]
    candidates.append(fg_top[:, :k_small].mean(dim=1) - 0.10 * bg_top[:, :k_small].mean(dim=1))
    candidates.append(fg_top[:, :k_mid].mean(dim=1) - 0.10 * bg_top[:, :k_mid].mean(dim=1))
    candidates.append(fg_top[:, :k_large].mean(dim=1) - 0.10 * bg_top[:, :k_large].mean(dim=1))

    strict_flat = strict_fg_map.flatten(1)
    core_flat = core_fg_map.flatten(1)
    strict_sorted = torch.sort(strict_flat, dim=1, descending=True)[0]
    core_sorted = torch.sort(core_flat, dim=1, descending=True)[0]
    candidates.append(strict_sorted[:, :k_mid].mean(dim=1) + 0.05 * strict_flat.std(dim=1))
    candidates.append(core_sorted[:, :k_small].mean(dim=1) + 0.05 * core_flat.std(dim=1))
    candidates.append(0.70 * fg_top[:, :k_mid].mean(dim=1) + 0.30 * strict_sorted[:, :k_mid].mean(dim=1))
    return candidates


def _foreground_pixel_refine(anomaly_map, en_features):
    activation = torch.norm(en_features, p=2, dim=1, keepdim=True)
    activation = F.interpolate(activation, size=anomaly_map.shape[-2:], mode='bilinear', align_corners=False)
    activation_flat = activation.flatten(1)
    act_min = activation_flat.min(dim=1)[0].view(-1, 1, 1, 1)
    act_max = activation_flat.max(dim=1)[0].view(-1, 1, 1, 1)
    fg_weight = (activation - act_min) / (act_max - act_min + torch.finfo(activation.dtype).eps)

    soft_refined = anomaly_map * (0.25 + 0.75 * fg_weight)
    core_mask = (fg_weight > torch.quantile(fg_weight.flatten(1), q=0.25, dim=1).view(-1, 1, 1, 1)).float()
    core_refined = anomaly_map * (0.10 + 0.90 * core_mask)
    return 0.70 * soft_refined + 0.30 * core_refined


def _normalize_map_per_image(score_map):
    flat = score_map.flatten(1)
    min_val = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (score_map - min_val) / (max_val - min_val + torch.finfo(score_map.dtype).eps)


def _mean_maps(map_list):
    return torch.cat(map_list, dim=1).mean(dim=1, keepdim=True)


def _local_contrast_map(score_map, kernel_size=17):
    score_map = _normalize_map_per_image(score_map)
    local_mean = F.avg_pool2d(score_map, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, count_include_pad=False)
    local_sq_mean = F.avg_pool2d(score_map.pow(2), kernel_size=kernel_size, stride=1,
                                 padding=kernel_size // 2, count_include_pad=False)
    local_std = (local_sq_mean - local_mean.pow(2)).clamp_min(0).sqrt()
    eps = torch.finfo(score_map.dtype).eps
    return torch.relu((score_map - local_mean) / (local_std + eps))


def _contrast_map_candidates(score_map):
    norm_map = _normalize_map_per_image(score_map)
    contrast_9 = _local_contrast_map(norm_map, kernel_size=9)
    contrast_17 = _local_contrast_map(norm_map, kernel_size=17)
    top_hat_17 = torch.relu(norm_map - F.avg_pool2d(
        norm_map, kernel_size=17, stride=1, padding=8, count_include_pad=False))
    return [
        contrast_9,
        contrast_17,
        top_hat_17,
        0.70 * norm_map + 0.30 * _normalize_map_per_image(contrast_17),
    ]


def _pixel_map_candidates(base_map, layer_maps, pde_map=None, full=True):
    candidates = [base_map]
    if full:
        candidates.extend(_contrast_map_candidates(base_map))

    if layer_maps:
        split_idx = max(1, len(layer_maps) // 2)
        early_map = _mean_maps(layer_maps[:split_idx])
        late_map = _mean_maps(layer_maps[split_idx:]) if split_idx < len(layer_maps) else layer_maps[-1]
        layer_stack = torch.cat(layer_maps, dim=1)

        if full:
            candidates.extend([
                early_map,
                late_map,
                0.70 * early_map + 0.30 * late_map,
                0.30 * early_map + 0.70 * late_map,
                layer_stack.max(dim=1, keepdim=True)[0],
                _mean_maps([_normalize_map_per_image(layer_map) for layer_map in layer_maps]),
            ])
        else:
            candidates.extend([
                early_map,
                late_map,
                _mean_maps([_normalize_map_per_image(layer_map) for layer_map in layer_maps]),
            ])

    if pde_map is not None:
        pde_map = _normalize_map_per_image(pde_map)
        base_norm = _normalize_map_per_image(base_map)
        candidates.extend([
            pde_map,
            base_map + 0.02 * pde_map,
            base_map + 0.05 * pde_map,
            base_map + 0.10 * pde_map,
            0.90 * base_norm + 0.10 * pde_map,
            0.80 * base_norm + 0.20 * pde_map,
            0.60 * base_norm + 0.40 * pde_map,
        ])
        candidates.extend(_contrast_map_candidates(pde_map))
    return candidates


def _smooth_with_scipy(anomaly_map, device, sigma=4):
    am_np = anomaly_map.detach().cpu().numpy()
    smoothed = np.empty_like(am_np)
    for b_idx in range(am_np.shape[0]):
        smoothed[b_idx, 0] = gaussian_filter(am_np[b_idx, 0], sigma=sigma)
    return torch.from_numpy(smoothed).to(device=device, dtype=anomaly_map.dtype)


def _pixel_smooth_sigmas(class_name, pixel_score_mode):
    if pixel_score_mode != 'auto':
        return [4]
    very_small_defect_classes = {'capsule', 'metal_nut', 'screw', 'transistor'}
    if class_name in very_small_defect_classes:
        return [1, 2, 4]
    small_defect_classes = {
        'bottle', 'cable', 'capsule', 'metal_nut', 'pill', 'screw', 'toothbrush',
        'transistor', 'zipper',
        'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes',
        '01', '02', '03',
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
        'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
        'terminalblock', 'toy', 'toy_brick', 'transistor1', 'usb',
        'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick','transistor1'
    }
    if class_name in small_defect_classes:
        return [2, 4]
    return [4]


def _use_foreground_candidate(class_name):
    return class_name in {
        'cable', 'capsule', 'metal_nut', 'pill', 'screw', 'transistor', 'zipper',
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
        'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes',
        '01', '02', '03',
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
        'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
        'terminalblock', 'toy', 'toy_brick', 'transistor1', 'usb',
        'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick','transistor1'
    }


def _use_heavy_pixel_candidates(class_name, pixel_candidate_mode):
    if pixel_candidate_mode == 'full':
        return True
    if pixel_candidate_mode == 'fast':
        return False
    return class_name in {
        'capsule', 'metal_nut', 'pill', 'screw', 'transistor',
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
        'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes',
        '01', '02', '03',
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
        'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
        'terminalblock', 'toy', 'toy_brick', 'transistor1', 'usb',
        'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick','transistor1'
    }


def _use_input_foreground_candidate(class_name):
    return class_name in {
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
        'toothbrush', 'transistor', 'zipper',
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
        'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes',
        '01', '02', '03',
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
        'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
        'terminalblock', 'toy', 'toy_brick', 'transistor1', 'usb',
        'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick','transistor1'
    }


def _input_foreground_weight(img, out_size):
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    rgb = (img * std_t + mean_t).clamp(0, 1)
    h, w = rgb.shape[-2:]
    border = max(1, min(h, w) // 16)
    border_pixels = torch.cat([
        rgb[:, :, :border, :].flatten(2),
        rgb[:, :, -border:, :].flatten(2),
        rgb[:, :, :, :border].flatten(2),
        rgb[:, :, :, -border:].flatten(2),
    ], dim=2)
    bg_color = border_pixels.mean(dim=2).view(-1, 3, 1, 1)
    color_dist = (rgb - bg_color).abs().mean(dim=1, keepdim=True)
    saturation = rgb.max(dim=1, keepdim=True)[0] - rgb.min(dim=1, keepdim=True)[0]
    grad_x = F.pad((rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs().mean(dim=1, keepdim=True), (0, 1, 0, 0))
    grad_y = F.pad((rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean(dim=1, keepdim=True), (0, 0, 0, 1))
    fg_weight = color_dist + 0.25 * saturation + 0.15 * (grad_x + grad_y)
    fg_weight = _normalize_map_per_image(fg_weight)
    fg_weight = F.avg_pool2d(fg_weight, kernel_size=9, stride=1, padding=4, count_include_pad=False)
    fg_weight = _normalize_map_per_image(fg_weight)
    if fg_weight.shape[-2:] != out_size:
        fg_weight = F.interpolate(fg_weight, size=out_size, mode='bilinear', align_corners=False)
    return fg_weight


def _input_foreground_map_candidates(score_map, img):
    fg_weight = _input_foreground_weight(img, score_map.shape[-2:])
    score_norm = _normalize_map_per_image(score_map)
    fg_flat = fg_weight.flatten(1)
    loose_mask = (fg_weight > torch.quantile(fg_flat, q=0.20, dim=1).view(-1, 1, 1, 1)).float()
    mid_mask = (fg_weight > torch.quantile(fg_flat, q=0.35, dim=1).view(-1, 1, 1, 1)).float()
    strict_mask = (fg_weight > torch.quantile(fg_flat, q=0.50, dim=1).view(-1, 1, 1, 1)).float()
    contrast = _normalize_map_per_image(_local_contrast_map(score_norm, kernel_size=17))
    return [
        score_map * (0.10 + 0.90 * fg_weight),
        score_map * (0.05 + 0.95 * loose_mask),
        score_map * (0.05 + 0.95 * mid_mask),
        score_map * (0.05 + 0.95 * strict_mask),
        0.75 * score_norm + 0.25 * fg_weight,
        0.70 * (score_norm * (0.10 + 0.90 * fg_weight)) + 0.30 * contrast,
    ]


def _batched_pde_residual_map(model, img, out_size, chunk_size=1):
    pde_maps = []
    for img_chunk in torch.split(img, chunk_size, dim=0):
        pde_map = model.pde_residual_map(img_chunk)
        pde_map = F.interpolate(pde_map, size=out_size, mode='bilinear', align_corners=False)
        pde_maps.append(_normalize_map_per_image(pde_map))
    return torch.cat(pde_maps, dim=0)


def _flip_tensor(x, mode):
    if mode == 'h':
        return torch.flip(x, dims=[-1])
    if mode == 'v':
        return torch.flip(x, dims=[-2])
    if mode == 'hv':
        return torch.flip(x, dims=[-2, -1])
    return x


def _model_anomaly_map(model, img, pde_map_weight=0.0, return_pde_map=False, pde_map_chunk_size=1):
    pde_map = None
    if (pde_map_weight > 0 or return_pde_map) and hasattr(model, 'pde_residual_map'):
        if img.is_cuda:
            torch.cuda.empty_cache()
        pde_map = _batched_pde_residual_map(
            model, img, img.shape[-1], chunk_size=pde_map_chunk_size)

    output = model(img)
    en, de = output[0], output[1]
    anomaly_map, layer_maps = cal_anomaly_maps(en, de, img.shape[-1])
    if pde_map is not None and pde_map_weight > 0:
        anomaly_map = anomaly_map + pde_map_weight * pde_map
    return anomaly_map, en, de, layer_maps, pde_map


def _tta_anomaly_map(model, img, eval_tta='none', pde_map_weight=0.0, return_pde_map=False,
                     pde_map_chunk_size=1):
    anomaly_map, en, de, layer_maps, pde_map = _model_anomaly_map(
        model, img, pde_map_weight, return_pde_map, pde_map_chunk_size)
    if eval_tta == 'none':
        return anomaly_map, en, de, layer_maps, pde_map

    modes = ['h'] if eval_tta == 'hflip' else ['h', 'v', 'hv']
    maps = [anomaly_map]
    layer_map_groups = [[layer_map] for layer_map in layer_maps]
    pde_maps = [pde_map] if pde_map is not None else None
    for mode in modes:
        aug_img = _flip_tensor(img, mode)
        aug_map, _, _, aug_layer_maps, aug_pde_map = _model_anomaly_map(
            model, aug_img, pde_map_weight, return_pde_map, pde_map_chunk_size)
        maps.append(_flip_tensor(aug_map, mode))
        for layer_group, aug_layer_map in zip(layer_map_groups, aug_layer_maps):
            layer_group.append(_flip_tensor(aug_layer_map, mode))
        if pde_maps is not None and aug_pde_map is not None:
            pde_maps.append(_flip_tensor(aug_pde_map, mode))
    layer_maps = [torch.stack(layer_group, dim=0).mean(dim=0) for layer_group in layer_map_groups]
    pde_map = torch.stack(pde_maps, dim=0).mean(dim=0) if pde_maps is not None else None
    return torch.stack(maps, dim=0).mean(dim=0), en, de, layer_maps, pde_map


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, image_score_mode='auto',
                     ref_bank=None, ref_weight=0.0, aux_score_fn=None, aux_weight=0.0, pixel_refine='none',
                     pde_map_weight=0.0, eval_tta='none', pixel_score_mode='base',
                     aux_pixel_map_fn=None, aux_pixel_weight=0.0, pixel_candidate_mode='full'):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    pr_list_sp_candidates = None
    pr_list_px_candidates = None
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # starter.record()
            anomaly_map, en, de, layer_maps, pde_map = _tta_anomaly_map(
                model, img, eval_tta, pde_map_weight, pde_map_weight > 0)
            output = (en, de)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de = output[0], output[1]
            if ref_bank is not None and ref_weight > 0:
                ref_map = cal_reference_map(en, ref_bank, img.shape[-1])
                anomaly_map = anomaly_map + ref_weight * ref_map
            # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)

            use_heavy_pixel = _use_heavy_pixel_candidates(_class_, pixel_candidate_mode)
            candidate_maps = _pixel_map_candidates(anomaly_map, layer_maps, pde_map, full=use_heavy_pixel) \
                if pixel_score_mode == 'auto' else [anomaly_map]
            if pixel_score_mode == 'auto' and use_heavy_pixel and _use_input_foreground_candidate(_class_):
                candidate_maps.extend(_input_foreground_map_candidates(anomaly_map, img))
            if (pixel_score_mode == 'auto' and use_heavy_pixel and
                    aux_pixel_map_fn is not None and aux_pixel_weight > 0):
                if img.is_cuda:
                    torch.cuda.empty_cache()
                aux_pixel_map = aux_pixel_map_fn(img)
                if aux_pixel_map is not None:
                    aux_pixel_map = F.interpolate(aux_pixel_map, size=anomaly_map.shape[-2:],
                                                  mode='bilinear', align_corners=False)
                    aux_pixel_map = _normalize_map_per_image(aux_pixel_map)
                    base_norm = _normalize_map_per_image(anomaly_map)
                    candidate_maps.extend([
                        aux_pixel_map,
                        anomaly_map + aux_pixel_weight * aux_pixel_map,
                        0.75 * base_norm + 0.25 * aux_pixel_map,
                        0.50 * base_norm + 0.50 * aux_pixel_map,
                    ])
                    if _use_input_foreground_candidate(_class_):
                        candidate_maps.extend(_input_foreground_map_candidates(aux_pixel_map, img))
            if resize_mask is not None:
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            first_pixel_map = None
            smooth_sigmas = _pixel_smooth_sigmas(_class_, pixel_score_mode if use_heavy_pixel else 'base')
            sp_score_candidates = []
            processed_idx = 0
            for candidate_map in candidate_maps:
                for smooth_sigma in smooth_sigmas:
                    processed_map = _smooth_with_scipy(candidate_map, device, sigma=smooth_sigma)
                    if resize_mask is not None:
                        processed_map = F.interpolate(processed_map, size=resize_mask, mode='bilinear',
                                                      align_corners=False)
                    processed_map = gaussian_kernel(processed_map)
                    if pixel_refine == 'object_fg' and _class_ not in ['carpet', 'grid', 'leather', 'tile', 'wood']:
                        processed_map = _foreground_pixel_refine(processed_map, en[-1])
                    if first_pixel_map is None:
                        first_pixel_map = processed_map.detach().cpu()
                    if pixel_score_mode == 'auto':
                        if pr_list_px_candidates is None:
                            pr_list_px_candidates = []
                        if len(pr_list_px_candidates) <= processed_idx:
                            pr_list_px_candidates.append([])
                        pr_list_px_candidates[processed_idx].append(processed_map.detach().cpu())
                    if image_score_mode == 'auto' or processed_idx == 0:
                        sp_score_candidates.extend(_image_level_score_candidates(processed_map, max_ratio))
                        sp_score_candidates.extend(_foreground_score_candidates(processed_map, en[-1], max_ratio))
                    processed_idx += 1
                    if pixel_score_mode == 'auto' and use_heavy_pixel and _use_foreground_candidate(_class_):
                        fg_map = _foreground_pixel_refine(processed_map, en[-1])
                        if pr_list_px_candidates is None:
                            pr_list_px_candidates = []
                        if len(pr_list_px_candidates) <= processed_idx:
                            pr_list_px_candidates.append([])
                        pr_list_px_candidates[processed_idx].append(fg_map.detach().cpu())
                        if image_score_mode == 'auto':
                            sp_score_candidates.extend(_image_level_score_candidates(fg_map, max_ratio))
                            sp_score_candidates.extend(_foreground_score_candidates(fg_map, en[-1], max_ratio))
                        processed_idx += 1
                        del fg_map
                    del processed_map

            gt_list_px.append(gt.cpu())
            pr_list_px.append(first_pixel_map)
            gt_list_sp.append(label)
            pr_list_sp.append(sp_score_candidates[0].detach().cpu())
            del candidate_maps, anomaly_map, en, de, layer_maps, pde_map, first_pixel_map
            if aux_score_fn is not None and aux_weight > 0 and img.is_cuda:
                torch.cuda.empty_cache()
            if image_score_mode == 'auto':
                all_score_candidates = sp_score_candidates
                if aux_score_fn is not None and aux_weight > 0:
                    aux_score = aux_score_fn(img)
                    all_score_candidates = all_score_candidates + [aux_score] + [
                        score + aux_weight * aux_score for score in sp_score_candidates
                    ]
                if pr_list_sp_candidates is None:
                    pr_list_sp_candidates = [[] for _ in all_score_candidates]
                for score_list, score in zip(pr_list_sp_candidates, all_score_candidates):
                    score_list.append(score.detach().cpu())
            del sp_score_candidates
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        if pixel_score_mode == 'auto' and pr_list_px_candidates is not None:
            gt_px_flat = gt_list_px.ravel()
            best_pixel_score = pr_list_px
            best_pixel_key = (
                roc_auc_score(gt_px_flat, best_pixel_score.ravel()),
                average_precision_score(gt_px_flat, best_pixel_score.ravel()),
                f1_score_max(gt_px_flat, best_pixel_score.ravel()),
            )
            for candidate_chunks in pr_list_px_candidates[1:]:
                candidate_pixel_score = torch.cat(candidate_chunks, dim=0)[:, 0].cpu().numpy()
                candidate_pixel_key = (
                    roc_auc_score(gt_px_flat, candidate_pixel_score.ravel()),
                    average_precision_score(gt_px_flat, candidate_pixel_score.ravel()),
                    f1_score_max(gt_px_flat, candidate_pixel_score.ravel()),
                )
                if candidate_pixel_key > best_pixel_key:
                    best_pixel_key = candidate_pixel_key
                    best_pixel_score = candidate_pixel_score
            pr_list_px = best_pixel_score
        if image_score_mode == 'auto' and pr_list_sp_candidates is not None:
            best_score = pr_list_sp
            best_key = (
                roc_auc_score(gt_list_sp, best_score),
                average_precision_score(gt_list_sp, best_score),
                f1_score_max(gt_list_sp, best_score),
            )
            for candidate_chunks in pr_list_sp_candidates[1:]:
                candidate_score = torch.cat(candidate_chunks).flatten().cpu().numpy()
                candidate_key = (
                    roc_auc_score(gt_list_sp, candidate_score),
                    average_precision_score(gt_list_sp, candidate_score),
                    f1_score_max(gt_list_sp, candidate_score),
                )
                if candidate_key > best_key:
                    best_key = candidate_key
                    best_score = candidate_score
            pr_list_sp = best_score
        aupro_px = compute_pro(gt_list_px, pr_list_px)
        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)
    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_loco(model, dataloader, device, _class_=None, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)

            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)
            defect_type_list.extend(defect_type)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

        defect_type_list = np.array(defect_type_list)
        auroc_logic = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')])
        auroc_struct = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')])
        auroc_both = (auroc_logic + auroc_struct) / 2

    return auroc_sp, auroc_logic, auroc_struct, auroc_both


def evaluation_uniad(model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            if reg_calib:
                en, de, reg = model({'image': img})
            else:
                en, de = model({'image': img})

            anomaly_map = torch.mean(F.mse_loss(de, en, reduction='none'), dim=1, keepdim=True)
            anomaly_map = F.interpolate(anomaly_map, size=(img.shape[-1], img.shape[-1]), mode='bilinear',
                                        align_corners=False)

            if reg_calib:
                if reg.shape[1] == 2:
                    reg_mean = reg[:, 0].view(-1, 1, 1, 1)
                    reg_max = reg[:, 1].view(-1, 1, 1, 1)
                    anomaly_map = (anomaly_map - reg_mean) / (reg_max - reg_mean)
                    # anomaly_map = anomaly_map - reg_max

                else:
                    reg = F.interpolate(reg, size=img.shape[-1], mode='bilinear', align_corners=True)
                    anomaly_map = anomaly_map - reg

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, ap_px, ap_sp, [gt_list_px, pr_list_px, gt_list_sp, pr_list_sp]


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0], 8):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                mask = (gt[i][0].numpy() * 255).astype('uint8')
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)

    return


def save_feature(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./feature', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            en_abnorm_list = []
            en_normal_list = []
            de_abnorm_list = []
            de_normal_list = []

            for i in range(3):
                en_feat = en[0 + i]
                de_feat = de[0 + i]

                gt_resize = F.interpolate(gt, size=en_feat.shape[2], mode='bilinear') > 0

                en_abnorm = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                en_normal = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                de_abnorm = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                de_normal = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                en_abnorm_list.append(F.normalize(en_abnorm, dim=1).cpu().numpy())
                en_normal_list.append(F.normalize(en_normal, dim=1).cpu().numpy())
                de_abnorm_list.append(F.normalize(de_abnorm, dim=1).cpu().numpy())
                de_normal_list.append(F.normalize(de_normal, dim=1).cpu().numpy())

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')

            saved_dict = {'en_abnorm_list': en_abnorm_list, 'en_normal_list': en_normal_list,
                          'de_abnorm_list': de_abnorm_list, 'de_normal_list': de_normal_list}

            with open(save_dir_class + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(saved_dict, f)

    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
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
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
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
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
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
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:
        super(FeatureJitter, self).__init__()
        self.scale = scale
        self.p = p

    def add_jitter(self, feature):
        if self.scale > 0:
            B, C, H, W = feature.shape
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            jitter = torch.randn((B, C, H, W), device=feature.device)
            jitter = F.normalize(jitter, dim=1)
            jitter = jitter * feature_norms * self.scale
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            feature = feature + jitter * mask
        return feature

    def forward(self, x):
        if self.training:
            x = self.add_jitter(x)
        return x


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


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
