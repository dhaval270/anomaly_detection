import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .features import build_extractor, extract_padim_features

D_REDUCED = 100     # random dimension selection
EPS = 0.01          # covariance regularization


class PaDiM:
    def __init__(self, d_reduced=D_REDUCED, device='cpu'):
        self.d_reduced = d_reduced
        self.device = device
        self.extractor = build_extractor(device)

        # Set after fit()
        self.selected_dims = None   # (d_reduced,)
        self.mean = None            # (H*W, d_reduced)
        self.precision = None       # (H*W, d_reduced, d_reduced)
        self.H = None
        self.W = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, dataloader):
        """Fit Gaussian at each spatial position from normal training images."""
        feats = extract_padim_features(self.extractor, dataloader, self.device)
        # feats: (N, 448, H, W)
        N, C, H, W = feats.shape
        self.H, self.W = H, W

        # Random dimension selection
        np.random.seed(42)
        self.selected_dims = np.random.choice(C, self.d_reduced, replace=False)
        feats = feats[:, self.selected_dims, :, :]  # (N, d_reduced, H, W)

        # Reshape to (H*W, N, d_reduced)
        feats = feats.permute(2, 3, 0, 1).reshape(H * W, N, self.d_reduced)
        feats_np = feats.numpy()   # (H*W, N, d_reduced)

        print('  Fitting Gaussians...', flush=True)
        n_patches = H * W
        self.mean = np.mean(feats_np, axis=1)                   # (H*W, d)
        centered = feats_np - self.mean[:, None, :]             # (H*W, N, d)
        cov = np.einsum('pni,pnj->pij', centered, centered) / (N - 1)  # (H*W, d, d)
        cov += EPS * np.eye(self.d_reduced)[None]               # regularise

        # Invert covariances in batch
        print('  Inverting covariances...', flush=True)
        self.precision = np.linalg.inv(cov)                     # (H*W, d, d)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, dataloader, corruption_fn=None):
        """
        Returns:
          image_scores : list[float]  - one per image (max of anomaly map)
          pixel_maps   : list[ndarray H×W]  - resized to IMG_SIZE×IMG_SIZE
          labels       : list[int]
          gt_masks     : list[ndarray or None]
        """
        from .dataset import IMG_SIZE

        image_scores, pixel_maps, labels, gt_masks = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='  Scoring', leave=False):
                imgs, lbl, mask, _ = batch
                if corruption_fn is not None:
                    # imgs is a batch of tensors; corruption applied at dataset level
                    pass
                imgs = imgs.to(self.device)
                f1, f2, f3 = self.extractor(imgs)
                H = f1.shape[2]
                f2_up = F.interpolate(f2, size=(H, H), mode='bilinear', align_corners=False)
                f3_up = F.interpolate(f3, size=(H, H), mode='bilinear', align_corners=False)
                feats = torch.cat([f1, f2_up, f3_up], dim=1)   # (B, 448, H, W)
                feats = feats[:, self.selected_dims, :, :]       # (B, d, H, W)

                B = imgs.shape[0]
                for b in range(B):
                    patch_feats = feats[b].cpu().numpy()         # (d, H, W)
                    patch_feats = patch_feats.reshape(self.d_reduced, -1).T  # (H*W, d)

                    delta = patch_feats - self.mean              # (H*W, d)
                    # Mahalanobis: sqrt(delta @ precision @ delta^T) per patch
                    tmp = np.einsum('pi,pij->pj', delta, self.precision)  # (H*W, d)
                    dist = np.sqrt(np.einsum('pi,pi->p', tmp, delta))      # (H*W,)
                    dist_map = dist.reshape(self.H, self.W)

                    # Smooth and resize to IMG_SIZE
                    dist_map = gaussian_filter(dist_map, sigma=4)
                    dist_map_resized = _resize_map(dist_map, IMG_SIZE)

                    image_scores.append(float(dist_map_resized.max()))
                    pixel_maps.append(dist_map_resized)
                    labels.append(int(lbl[b]))
                    gt_masks.append(mask[b].numpy() if hasattr(mask[b], 'numpy') else mask[b])

        return image_scores, pixel_maps, labels, gt_masks


# ── Helper ────────────────────────────────────────────────────────────────────

def _resize_map(arr, size):
    from PIL import Image
    img = Image.fromarray(arr.astype(np.float32))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)
