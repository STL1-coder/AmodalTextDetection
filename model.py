import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
try:
    import timm
except ImportError:
    pass

class PyramidalGFPBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0
        group_width = channels // 4
        self.conv1 = nn.Conv2d(group_width, group_width, 3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(group_width, group_width, 3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(group_width, group_width, 3, padding=3, dilation=3, bias=False)
        self.conv4 = nn.Conv2d(group_width, group_width, 3, padding=5, dilation=5, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.fusion = nn.Conv2d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        y4 = self.conv4(x4)
        out = torch.cat([y1, y2, y3, y4], dim=1)
        out = self.act(self.bn(out))
        out = self.fusion(out)
        out = self.dropout(out)
        return out + x

class MIMViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224.mae', out_channels=256):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        self.patch_size = 16
        self.embed_dim = self.vit.embed_dim
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_features = self.vit.forward_features(x)
        x_features = x_features[:, 1:, :]
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size
        x_spatial = x_features.transpose(1, 2).reshape(B, self.embed_dim, h_feat, w_feat)
        return self.proj(x_spatial)

class CRFTLayer(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_c2m = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.attn_m2c = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        self.norm_mlp = nn.LayerNorm(channels)

    def forward(self, cnn_feat, vit_feat):
        B, C, H, W = cnn_feat.shape
        f_c = cnn_feat.flatten(2).transpose(1, 2)
        f_v = vit_feat.flatten(2).transpose(1, 2)
        q_c = self.norm1(f_c)
        out_c, _ = self.attn_c2m(q_c, f_v, f_v)
        q_v = self.norm2(f_v)
        out_v, _ = self.attn_m2c(q_v, f_c, f_c)
        fused = out_c + out_v + f_c
        fused = fused + self.mlp(self.norm_mlp(fused))
        return fused.transpose(1, 2).reshape(B, C, H, W)

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, scale_factor=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.final = nn.Conv2d(in_channels // 2, 1, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.final(x)

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPNFusion(nn.Module):
    def __init__(self, channels=[64, 128, 256], out_channels=128):
        super().__init__()
        self.lat_l3 = nn.Conv2d(channels[2], out_channels, 1)
        self.lat_l2 = nn.Conv2d(channels[1], out_channels, 1)
        self.lat_l1 = nn.Conv2d(channels[0], out_channels, 1)
        self.smooth_l3 = ConvBnRelu(out_channels, out_channels)
        self.smooth_l2 = ConvBnRelu(out_channels, out_channels)
        self.smooth_l1 = ConvBnRelu(out_channels, out_channels)

    def forward(self, l1, l2, l3):
        p3 = self.lat_l3(l3)
        p3_smooth = self.smooth_l3(p3)
        p3_up = F.interpolate(p3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lat_l2(l2) + p3_up
        p2_smooth = self.smooth_l2(p2)
        p2_up = F.interpolate(p2, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.lat_l1(l1) + p2_up
        p1_smooth = self.smooth_l1(p1)
        return p1_smooth

class OcclusionAwareDetector(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.mim_vit = MIMViTEncoder('vit_base_patch16_224.mae', out_channels=256)
        self.crft = CRFTLayer(256, dropout=0.1)
        self.fpn = FPNFusion(channels=[64, 128, 256], out_channels=128)
        self.head = nn.Sequential(
            ConvBnRelu(128, 64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x_stem = self.stem(x)
        l1 = self.layer1(x_stem)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        v = self.mim_vit(x)
        if v.shape[-2:] != l3.shape[-2:]:
            v = F.interpolate(v, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        l3_enhanced = self.crft(l3, v)
        fpn_feat = self.fpn(l1, l2, l3_enhanced)
        logits = self.head(fpn_feat)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        return logits
