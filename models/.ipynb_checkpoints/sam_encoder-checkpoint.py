import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class SAMEncoder(nn.Module):
    def __init__(self, 
                 model_type="vit_b", 
                 checkpoint_path="./sam_vit_b_01ec64.pth", 
                 out_dim=768): 
        super().__init__()
        self.is_sam = True  
        
        print(f"Loading Multi-scale SAM ({model_type})...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam.image_encoder
        
        # 冻结 SAM 的权重
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # 【核心修正】：SAM 每层特征是 256 维。
        # 我们要提取 3 层，拼接后就是 256 * 3 = 768 维。
        # 所以这里的输入通道强制写死为 768。
        self.adapter = nn.Sequential(
            nn.Conv2d(2304, out_dim, kernel_size=1), 
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 瞒天过海：放大到 SAM 需要的 1024
        x_sam = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        layer_features = []
        # 【核心修正】：严格限定只提取第 4, 8, 12 个 Block (索引 3, 7, 11)
        select_layers = [3, 7, 11] 
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feat = self.image_encoder.patch_embed(x_sam)
                if self.image_encoder.pos_embed is not None:
                    feat = feat + self.image_encoder.pos_embed

                for i, blk in enumerate(self.image_encoder.blocks):
                    feat = blk(feat)
                    # 只在这 3 层拦截特征
                    if i in select_layers:
                        p_feat = feat.permute(0, 3, 1, 2).contiguous()
                        layer_features.append(p_feat)
                
                # 将这 3 层特征拼接在一起 (3个256 -> 变成 768 维)
                multi_feat = torch.cat(layer_features, dim=1)
                
        # 转回 float32 并缩小到 Dinomaly 需要的分辨率
        multi_feat = multi_feat.float()
        #target_H, target_W = H // 16, W // 16
        #multi_feat = F.interpolate(multi_feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
        target_H, target_W = 48, 48
        # 通过 Adapter (768 进，out_dim 出)
        out = self.adapter(multi_feat) 
        #out = self.adapter(multi_feat)
        out = F.interpolate(out, size=(target_H, target_W), mode='bilinear', align_corners=False)
        # 展平为序列返回
        out = out.flatten(2).transpose(1, 2) 
        return (out, )