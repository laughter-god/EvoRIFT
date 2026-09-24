import os
import random
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score

# ================= 自定义模块导入 =================
from dataset import get_data_transforms, get_strong_transforms, MVTecDataset
from models.sam_encoder import SAMEncoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dinov1.utils import trunc_normal_
from optimizers import StableAdamW
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler

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
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(item_list, args, print_fn):
    setup_seed(1)
    
    batch_size = 8
    image_size = 400
    crop_size = 384
    device = args.device

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

        # MBDDDataset / MVTecDataset 均可兼容此路径结构
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # ==========================================
    # 🌟 修复点 1：动态计算 Iterations，适配任何数据集大小！
    # ==========================================
    total_epochs = 24
    eval_interval = 4  # 缩短评估间隔，方便同时监控两个数据集
    total_iters = len(train_dataloader) * total_epochs

    embed_dim = 768
    num_heads = 12
    encoder = SAMEncoder(
        model_type="vit_b", 
        checkpoint_path="./sam_vit_b_01ec64.pth", 
        out_dim=embed_dim
    )

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2) 
        for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder, 
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9], mask_neighbor_size=0, 
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]], fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]]
    ).to(device)

    trainable = nn.ModuleList([bottleneck, decoder, model.defect_injector, model.pde_layer])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    
    # ==========================================
    # 🌟 修复点 2：使用计算好的 total_iters 初始化调度器
    # ==========================================
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters, warmup_iters=100)

    print_fn(f'Train image number: {len(train_data)}')
    print_fn(f'Total Iterations: {total_iters} | Total Epochs: {total_epochs}')

    it = 0
    for epoch in range(total_epochs):
        model.train()
        loss_list = []
        current_epoch = epoch + 1
        
        pbar = tqdm(train_dataloader, desc=f'Epoch [{current_epoch}/{total_epochs}]', leave=False, ncols=None, bar_format='{desc} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        for img, label in pbar:
            img, label = img.to(device), label.to(device)

            # ==========================================
            # 🌟 修复点 3：三变量接收，接通标签制导与 PDE 正交火力！
            # ==========================================
            en, de, ortho_loss = model(img, label=label)

            # --- 双路语义最大化探针 (Dual-Region Max Probe) ---
            with torch.no_grad():
                cos_sim_map = torch.nn.functional.cosine_similarity(en[-1], de[-1], dim=1)
                error_map = 1.0 - cos_sim_map 
                B, H, W = error_map.shape
                
                sam_activation = torch.norm(en[-1], p=2, dim=1)
                bg_mask = (sam_activation < sam_activation.mean(dim=[1, 2], keepdim=True)).float()
                
                error_map_flat, bg_mask_flat = error_map.view(B, -1), bg_mask.view(B, -1)
                
                std_max_list = []
                for i in range(B):
                    bg_errors = error_map_flat[i][bg_mask_flat[i] == 1]
                    fg_errors = error_map_flat[i][bg_mask_flat[i] == 0]
                    std_bg = torch.std(bg_errors).item() if len(bg_errors) > (H * W * 0.05) else 0.0
                    std_fg = torch.std(fg_errors).item() if len(fg_errors) > (H * W * 0.05) else 0.0
                    std_max_list.append(max(std_bg, std_fg))
                    
                std_val_final = np.mean(std_max_list)
            
            p_adaptive = max(0.95 - 2.0 * std_val_final, 0.5)
            # --------------------------------------------------

            # 融合主干重建 Loss 与 PDE 正交解耦约束
            loss = global_cosine_hm_percent(en, de, p=p_adaptive, factor=0.1)


            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            
            current_loss = loss.item()
            loss_list.append(current_loss)
            lr_scheduler.step()

            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
            it += 1
                
        print_fn('Epoch [{}/{}], Avg Loss: {:.4f}'.format(current_epoch, total_epochs, np.mean(loss_list)))

        if current_epoch % eval_interval == 0 or current_epoch == total_epochs:
            print_fn('\n' + '='*15 + ' 开始第 {} 轮性能评估 '.format(current_epoch) + '='*15)
            
            auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
            auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

            for item, test_data in zip(item_list, test_data_list):
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
                
                # 🌟 调用你全新整合的大一统 evaluation_batch
               # 改为原分辨率 384，且把 max_ratio 调到大一统黄金比例 0.005
                results = evaluation_batch(model, test_dataloader, device, max_ratio=0.005, resize_mask=384)
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                auroc_sp_list.append(auroc_sp); ap_sp_list.append(ap_sp); f1_sp_list.append(f1_sp)
                auroc_px_list.append(auroc_px); ap_px_list.append(ap_px); f1_px_list.append(f1_px); aupro_px_list.append(aupro_px)

                print_fn('  ➤ {}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f} | P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item.ljust(12), auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

            print_fn('-'*90)
            print_fn(' Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f} | P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
            print_fn('='*52 + '\n')

            model.train()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='Unified Training for MVTec and VisA')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset (e.g., /root/autodl-tmp/MVTecAD/data/mvtec)')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    args = parser.parse_args()

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ==========================================
    # 🌟 动态路由：根据路径自动切换类别列表和存档名
    # ==========================================
    if 'visa' in args.data_path.lower():
        dataset_name = 'VisA'
        item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                     'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    else:
        dataset_name = 'MVTec'
        item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                     'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    args.save_name = f'vitill_{dataset_name.lower()}_unified_run'
    
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    print_fn(f"Detected Dataset: {dataset_name}")
    print_fn(f"Target Device: {args.device}")
    print_fn(f"Data Path: {args.data_path}")

    train(item_list, args, print_fn)