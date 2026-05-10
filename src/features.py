"""Pretrained ResNet18 feature extractor for PaDiM and PatchCore."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


class ResNet18Features(nn.Module):
    """Extract intermediate features from ResNet18 (layer1, layer2, layer3)."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Stem
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1   # output: (B, 64,  H/4,  W/4)  -> 56×56
        self.layer2 = resnet.layer2   # output: (B, 128, H/8,  W/8)  -> 28×28
        self.layer3 = resnet.layer3   # output: (B, 256, H/16, W/16) -> 14×14

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


def build_extractor(device='cpu'):
    model = ResNet18Features().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── Spatial sizes for 224×224 input ──────────────────────────────────────────
# layer1 -> 56×56, layer2 -> 28×28, layer3 -> 14×14

PADIM_TARGET_SIZE = 56   # upsample everything to layer1 resolution
PATCHCORE_TARGET_SIZE = 28  # upsample layer3 to layer2 resolution


# ── PaDiM features ────────────────────────────────────────────────────────────

def extract_padim_features(extractor, dataloader, device='cpu',
                           corruption_fn=None):
    """
    Returns tensor of shape (N, C_total, H, W) where:
      C_total = 64 + 128 + 256 = 448  (after upsampling to 56×56)
    """
    all_feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting PaDiM features', leave=False):
            imgs = batch[0].to(device)
            f1, f2, f3 = extractor(imgs)
            H = f1.shape[2]  # 56
            f2_up = F.interpolate(f2, size=(H, H), mode='bilinear', align_corners=False)
            f3_up = F.interpolate(f3, size=(H, H), mode='bilinear', align_corners=False)
            feats = torch.cat([f1, f2_up, f3_up], dim=1)   # (B, 448, 56, 56)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)   # (N, 448, 56, 56)



def extract_patchcore_features(extractor, dataloader, device='cpu'):
    
    all_feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting PatchCore features', leave=False):
            imgs = batch[0].to(device)
            _, f2, f3 = extractor(imgs)
            H = f2.shape[2]  # 28
            f3_up = F.interpolate(f3, size=(H, H), mode='bilinear', align_corners=False)
            feats = torch.cat([f2, f3_up], dim=1)   # (B, 384, 28, 28)
            # Locally aware patch features: 3×3 neighbourhood average
            feats = F.avg_pool2d(feats, kernel_size=3, stride=1, padding=1)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)   # (N, 384, 28, 28)
