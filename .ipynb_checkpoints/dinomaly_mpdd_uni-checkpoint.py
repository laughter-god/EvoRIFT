# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tqdm import tqdm
import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import math
from torch.utils.data import DataLoader, ConcatDataset
from models.sam_encoder import SAMEncoder
from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import BTADDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    regional_cosine_focal, WarmCosineScheduler
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from pytorch_msssim import ssim

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'mpdd.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_to_dino_patch_multiple(img, patch_size=14):
    h, w = img.shape[-2:]
    target_h = int(math.ceil(h / patch_size) * patch_size)
    target_w = int(math.ceil(w / patch_size) * patch_size)
    if target_h == h and target_w == w:
        return img
    return F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)


def extract_dino_image_features(encoder, img, target_layer=9):
    img = resize_to_dino_patch_multiple(img)
    x = encoder.prepare_tokens(img)
    target_feat = None
    for i, blk in enumerate(encoder.blocks):
        x = blk(x)
        if i == target_layer:
            target_feat = x
            break
    if target_feat is None:
        target_feat = x

    num_register_tokens = getattr(encoder, 'num_register_tokens', 0)
    patch_tokens = target_feat[:, 1 + num_register_tokens:, :]
    image_feat = patch_tokens.mean(dim=1)
    return F.normalize(image_feat, dim=1)


def build_dino_image_bank(encoder, dataloader, device):
    encoder.eval()
    bank = []
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            bank.append(extract_dino_image_features(encoder, img).cpu())
    if not bank:
        return None
    return torch.cat(bank, dim=0).to(device)


def extract_dino_patch_features(encoder, img, target_layer=9):
    img = resize_to_dino_patch_multiple(img)
    x = encoder.prepare_tokens(img)
    target_feat = None
    for i, blk in enumerate(encoder.blocks):
        x = blk(x)
        if i == target_layer:
            target_feat = x
            break
    if target_feat is None:
        target_feat = x

    num_register_tokens = getattr(encoder, 'num_register_tokens', 0)
    patch_tokens = target_feat[:, 1 + num_register_tokens:, :]
    return F.normalize(patch_tokens, dim=-1)


def build_dino_patch_bank(encoder, dataloader, device, max_patches=20000):
    encoder.eval()
    patch_bank = []
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            patches = extract_dino_patch_features(encoder, img)
            patches = patches.reshape(-1, patches.shape[-1])
            patch_bank.append(patches.cpu())
            if sum(p.shape[0] for p in patch_bank) > max_patches * 2:
                patch_bank = [torch.cat(patch_bank, dim=0)]
                keep_idx = torch.randperm(patch_bank[0].shape[0])[:max_patches]
                patch_bank = [patch_bank[0][keep_idx]]
    if not patch_bank:
        return None
    patch_bank = torch.cat(patch_bank, dim=0)
    if patch_bank.shape[0] > max_patches:
        keep_idx = torch.randperm(patch_bank.shape[0])[:max_patches]
        patch_bank = patch_bank[keep_idx]
    return patch_bank.to(device)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def dino_patch_multi_score(encoder, img, patch_bank, top_ratios, chunk_size=2048):
    patches = extract_dino_patch_features(encoder, img)
    batch_size, num_patches, feat_dim = patches.shape
    patches = patches.reshape(-1, feat_dim)
    distances = []
    with torch.no_grad():
        for chunk in patches.split(chunk_size, dim=0):
            nearest_sim = torch.matmul(chunk, patch_bank.t()).max(dim=1)[0]
            distances.append(1.0 - nearest_sim)
    distances = torch.cat(distances, dim=0).reshape(batch_size, num_patches)
    sorted_distances = torch.sort(distances, dim=1, descending=True)[0]

    scores = []
    for ratio in top_ratios:
        k = max(1, int(num_patches * ratio))
        scores.append(sorted_distances[:, :k].mean(dim=1))
    return torch.stack(scores, dim=1)


def dino_patch_distance_map(encoder, img, patch_bank, out_size, chunk_size=2048):
    patches = extract_dino_patch_features(encoder, img)
    batch_size, num_patches, feat_dim = patches.shape
    patches = patches.reshape(-1, feat_dim)
    distances = []
    with torch.no_grad():
        for chunk in patches.split(chunk_size, dim=0):
            nearest_sim = torch.matmul(chunk, patch_bank.t()).max(dim=1)[0]
            distances.append(1.0 - nearest_sim)
    distances = torch.cat(distances, dim=0).reshape(batch_size, num_patches)
    side = int(math.sqrt(num_patches))
    distance_map = distances.view(batch_size, 1, side, side)
    distance_map = F.interpolate(distance_map, size=(out_size, out_size), mode='bilinear', align_corners=False)
    flat = distance_map.flatten(1)
    min_val = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return (distance_map - min_val) / (max_val - min_val + torch.finfo(distance_map.dtype).eps)


def build_dino_patch_score_stats(encoder, dataloader, patch_bank, device, top_ratios):
    score_list = []
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            score_list.append(dino_patch_multi_score(encoder, img, patch_bank, top_ratios).cpu())
    scores = torch.cat(score_list, dim=0)
    center = torch.quantile(scores, 0.50, dim=0)
    spread = torch.quantile(scores, 0.75, dim=0) - torch.quantile(scores, 0.25, dim=0)
    spread = spread.clamp_min(1e-3)
    return {
        'center': center.to(device),
        'spread': spread.to(device),
    }


def calibrated_dino_patch_score(encoder, img, patch_bank, score_stats, top_ratios):
    scores = dino_patch_multi_score(encoder, img, patch_bank, top_ratios)
    z_scores = (scores - score_stats['center']) / score_stats['spread']
    return torch.relu(z_scores).max(dim=1)[0]


def batched_pde_residual_metrics(model, img, chunk_size=1):
    if img.is_cuda:
        torch.cuda.empty_cache()
    metric_chunks = []
    for img_chunk in torch.split(img, chunk_size, dim=0):
        metric_chunks.append(model.pde_residual_metrics(img_chunk))
    return torch.cat(metric_chunks, dim=0)


def build_pde_metric_stats(model, dataloader, device):
    model.eval()
    metric_list = []
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            metric_list.append(batched_pde_residual_metrics(model, img).detach().cpu())
    metrics = torch.cat(metric_list, dim=0)
    center = torch.quantile(metrics, 0.50, dim=0)
    spread = torch.quantile(metrics, 0.75, dim=0) - torch.quantile(metrics, 0.25, dim=0)
    spread = spread.clamp_min(1e-3)
    return {
        'center': center.to(device),
        'spread': spread.to(device),
    }


def calibrated_pde_metric_score(model, img, metric_stats):
    metrics = batched_pde_residual_metrics(model, img)
    z_scores = (metrics - metric_stats['center']) / metric_stats['spread']
    return torch.relu(z_scores).max(dim=1)[0]


def train(item_list):
    setup_seed(1)


    batch_size = 8
    image_size = 448
    crop_size = 392

    # image_size = 448
    # crop_size = 448

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = BTADDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    encoder_name = 'dinov2reg_vit_base_14'
    total_epochs = 24
    eval_interval = 12
    total_iters = len(train_dataloader) * total_epochs
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim = 768  # 设定好维度
    num_heads = 12
    encoder = SAMEncoder(
        model_type="vit_b", 
        checkpoint_path="./sam_vit_b_01ec64.pth", # 请确保权重文件路径正确
        out_dim=embed_dim
        )

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    # bottleneck.append(nn.Sequential(FeatureJitter(scale=40),
    #                                 bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)))

    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        # blk = ConvBlock(dim=embed_dim, kernel_size=7, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    #trainable = nn.ModuleList([bottleneck, decoder])
   # trainable = nn.ModuleList([bottleneck, decoder, model.defect_injector])
    trainable = nn.ModuleList([bottleneck, decoder, model.defect_injector, model.pde_layer])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)
    use_amp = False
    scaler = GradScaler(enabled=use_amp)
    dino_aux_encoder = None
    if args.dino_aux_weight > 0 or args.dino_patch_weight > 0 or args.dino_pixel_weight > 0:
        print_fn('DINO aux encoder enabled, image_weight={}, patch_weight={}, pixel_weight={}'.format(
            args.dino_aux_weight, args.dino_patch_weight, args.dino_pixel_weight))
        dino_aux_encoder = vit_encoder.load(args.dino_aux_encoder).to(device)
        dino_aux_encoder.eval()
        for param in dino_aux_encoder.parameters():
            param.requires_grad = False
    if args.pde_gate_weight > 0:
        print_fn('PDE residual metric gate enabled, weight={}'.format(args.pde_gate_weight))

    print_fn('train image number:{}'.format(len(train_data)))

   # ==========================================
    # 🌟 1. 设定总轮数和评估频率
    # ==========================================


    it = 0  # 保留全局迭代计数器，为了算你那个动态的 p 值
    
    for epoch in range(total_epochs):
        model.train()
        loss_list = []
        current_epoch = epoch + 1
        
        # ==========================================
        # 🌟 2. 召唤进度条！
        # desc: 前缀显示第几轮 | leave=False: 跑完一轮后进度条自动消失，不刷屏终端 | ncols: 进度条宽度
        # ==========================================
        pbar = tqdm(train_dataloader, desc=f'Epoch [{current_epoch}/{total_epochs}]', leave=False, ncols=None,bar_format='{desc} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        for img, label in pbar:
            img = img.to(device)
            label = label.to(device)

            with autocast(enabled=use_amp):
                en, de = model(img)

            # 动态 p 值计算逻辑保持不变
# ==========================================
            # 🌟 终极融合版：双路语义最大化探针 (Dual-Region Max Probe)
            # ==========================================
            with torch.no_grad():

                cos_sim_map = torch.nn.functional.cosine_similarity(en[-1], de[-1], dim=1)

                error_map = 1.0 - cos_sim_map  # [B, H, W]

                B, H, W = error_map.shape

                

                sam_activation = torch.norm(en[-1], p=2, dim=1)

                bg_mask = (sam_activation < sam_activation.mean(dim=[1, 2], keepdim=True)).float()

                

                error_map_flat = error_map.view(B, -1)

                bg_mask_flat = bg_mask.view(B, -1)

                

                std_max_list = []

                for i in range(B):

                    # 1. 分离出背景误差和前景误差

                    bg_errors = error_map_flat[i][bg_mask_flat[i] == 1]

                    fg_errors = error_map_flat[i][bg_mask_flat[i] == 0]

                    

                    # 2. 分别计算两个区域的独立波动 (保底防止无前景/无背景的情况)

                    std_bg = torch.std(bg_errors).item() if len(bg_errors) > (H * W * 0.05) else 0.0

                    std_fg = torch.std(fg_errors).item() if len(fg_errors) > (H * W * 0.05) else 0.0

                    

                    # 3. 核心物理逻辑：不管子弹打在背景还是前景，只要有一端炸了，就取最大值拉响警报！

                    std_max_list.append(max(std_bg, std_fg))

                

                # 获取全图最危险区域的真实波动

                std_val_final = np.mean(std_max_list)



            # 4. 魔法演化：用双路探针捕捉到的最大激波来压迫网络

            if not np.isfinite(std_val_final):
                std_val_final = 0.0
            p_adaptive = float(np.clip(args.p_base - args.p_std_scale * std_val_final, args.p_min, args.p_max))
            # ==========================================
           # p = min(p_adaptive * it / 1000, p_adaptive)
            # 3. 把这张图专属的 p 喂给原版 Loss！
            with autocast(enabled=use_amp):
                loss_cos = global_cosine_hm_percent(en, de, p=p_adaptive, factor=0.1)
                loss_ssim = 1 - ssim(de[-1], en[-1], data_range=1, size_average=True)
                loss = loss_cos + args.ssim_weight * loss_ssim
                if args.regional_focal_weight > 0:
                    loss_regional = regional_cosine_focal(en, de, p=args.regional_focal_p,
                                                          alpha=args.regional_focal_alpha)
                    loss = loss + args.regional_focal_weight * loss_regional

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            scaler.step(optimizer)
            scaler.update()
            
            current_loss = loss.item()
            loss_list.append(current_loss)
            lr_scheduler.step()

            # 🌟 3. 在进度条尾部实时更新当前的 Loss 和 p 值，极度解压！
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

            it += 1
                
        # 这一轮的进度条跑完后，打印本轮的平均 Loss
        print_fn('Epoch [{}/{}], Avg Loss: {:.4f}'.format(
            current_epoch, total_epochs, np.mean(loss_list)))

        # ==========================================
        # 🌟 4. 定期评估（UI排版优化版）
        # ==========================================
        if current_epoch % eval_interval == 0 or current_epoch == total_epochs:
            print_fn('\n' + '='*15 + ' 开始第 {} 轮性能评估 '.format(current_epoch) + '='*15)
            
            auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
            auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

            model.eval()
            eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else batch_size
            for item, train_data_for_item, test_data in zip(item_list, train_data_list, test_data_list):
                test_dataloader = torch.utils.data.DataLoader(
                    test_data, batch_size=eval_batch_size, shuffle=False, num_workers=4)
                aux_score_parts = []
                aux_pixel_map_fn = None
                aux_train_dataloader = None
                if (args.dino_aux_weight > 0 or args.dino_patch_weight > 0 or
                        args.dino_pixel_weight > 0 or args.pde_gate_weight > 0):
                    aux_train_dataloader = torch.utils.data.DataLoader(
                        train_data_for_item, batch_size=eval_batch_size, shuffle=False, num_workers=4)

                if dino_aux_encoder is not None and args.dino_aux_weight > 0:
                    dino_bank = build_dino_image_bank(dino_aux_encoder, aux_train_dataloader, device)

                    def dino_score_fn(img, bank=dino_bank):
                        with torch.no_grad():
                            feat = extract_dino_image_features(dino_aux_encoder, img)
                            return 1.0 - torch.matmul(feat, bank.t()).max(dim=1)[0]

                    aux_score_parts.append((args.dino_aux_weight, dino_score_fn))

                if dino_aux_encoder is not None and (args.dino_patch_weight > 0 or args.dino_pixel_weight > 0):
                    dino_patch_bank = build_dino_patch_bank(
                        dino_aux_encoder, aux_train_dataloader, device,
                        max_patches=args.dino_patch_bank_size)
                    if args.dino_patch_weight > 0:
                        dino_patch_ratios = parse_float_list(args.dino_patch_top_ratios)
                        dino_patch_stats = build_dino_patch_score_stats(
                            dino_aux_encoder, aux_train_dataloader, dino_patch_bank, device, dino_patch_ratios)

                        def dino_patch_score_fn(img, bank=dino_patch_bank, stats=dino_patch_stats,
                                                ratios=dino_patch_ratios):
                            with torch.no_grad():
                                return calibrated_dino_patch_score(
                                    dino_aux_encoder, img, bank, stats, ratios)

                        aux_score_parts.append((args.dino_patch_weight, dino_patch_score_fn))

                    if args.dino_pixel_weight > 0:
                        def aux_pixel_map_fn(img, bank=dino_patch_bank):
                            with torch.no_grad():
                                return dino_patch_distance_map(
                                    dino_aux_encoder, img, bank, out_size=img.shape[-1])

                if args.pde_gate_weight > 0:
                    pde_metric_stats = build_pde_metric_stats(model, aux_train_dataloader, device)

                    def pde_score_fn(img, stats=pde_metric_stats):
                        with torch.no_grad():
                            return calibrated_pde_metric_score(model, img, stats)

                    aux_score_parts.append((args.pde_gate_weight, pde_score_fn))

                aux_score_fn = None
                if aux_score_parts:
                    def aux_score_fn(img, parts=aux_score_parts):
                        score = None
                        for weight, score_fn in parts:
                            part_score = weight * score_fn(img)
                            score = part_score if score is None else score + part_score
                        return score

                candidate_class = args.mpdd_candidate_class if args.mpdd_candidate_class else item
                results = evaluation_batch(
                    model, test_dataloader, device, _class_=candidate_class, max_ratio=args.eval_max_ratio,
                    resize_mask=args.eval_resize_mask,
                    image_score_mode='auto', aux_score_fn=aux_score_fn, aux_weight=1.0,
                    pixel_refine=args.pixel_refine,
                    pde_map_weight=args.pde_map_weight, eval_tta=args.eval_tta,
                    pixel_score_mode=args.pixel_score_mode,
                    aux_pixel_map_fn=aux_pixel_map_fn, aux_pixel_weight=args.dino_pixel_weight,
                    pixel_candidate_mode=args.pixel_candidate_mode)
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                auroc_sp_list.append(auroc_sp)
                ap_sp_list.append(ap_sp)
                f1_sp_list.append(f1_sp)
                auroc_px_list.append(auroc_px)
                ap_px_list.append(ap_px)
                f1_px_list.append(f1_px)
                aupro_px_list.append(aupro_px)

                # 打印单个类别的成绩，使用 ljust 对齐，强迫症狂喜
                print_fn('  ➤ {}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f} | P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item.ljust(12), auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 打印 Mean 均分
            print_fn('-'*90)
            print_fn(' Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f} | P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
            print_fn('='*52 + '\n')

            model.train() # 评估完切回训练模式

    return

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../btad')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='mpdd')
    parser.add_argument('--dino_aux_weight', type=float, default=0.2)
    parser.add_argument('--dino_aux_encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--dino_patch_weight', type=float, default=0.2)
    parser.add_argument('--dino_pixel_weight', type=float, default=0.0)
    parser.add_argument('--dino_patch_top_ratios', type=str, default='0.005,0.01,0.02')
    parser.add_argument('--dino_patch_bank_size', type=int, default=20000)
    parser.add_argument('--pde_gate_weight', type=float, default=0.5)
    parser.add_argument('--pixel_refine', type=str, default='none', choices=['none', 'object_fg'])
    parser.add_argument('--pde_map_weight', type=float, default=0.0)
    parser.add_argument('--eval_tta', type=str, default='none', choices=['none', 'hflip', 'hvflip'])
    parser.add_argument('--eval_batch_size', type=int, default=0)
    parser.add_argument('--eval_resize_mask', type=int, default=256)
    parser.add_argument('--eval_max_ratio', type=float, default=0.01)
    parser.add_argument('--pixel_score_mode', type=str, default='auto', choices=['base', 'auto'])
    parser.add_argument('--pixel_candidate_mode', type=str, default='full', choices=['fast', 'weak', 'full'])
    parser.add_argument('--mpdd_candidate_class', type=str, default='')
    parser.add_argument('--p_base', type=float, default=0.95)
    parser.add_argument('--p_std_scale', type=float, default=2.0)
    parser.add_argument('--p_min', type=float, default=0.75)
    parser.add_argument('--p_max', type=float, default=0.95)
    parser.add_argument('--ssim_weight', type=float, default=0.06)
    parser.add_argument('--regional_focal_weight', type=float, default=0.02)
    parser.add_argument('--regional_focal_p', type=float, default=0.98)
    parser.add_argument('--regional_focal_alpha', type=float, default=2.0)
    args = parser.parse_args()
    #
    item_list = ['bracket_black', 'bracket_brown', 'bracket_white','connector','metal_plate','tubes']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
