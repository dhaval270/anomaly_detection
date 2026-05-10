
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .features import build_extractor, extract_patchcore_features

CORESET_RATIO = 0.01    # keep 1 % of patches
K_NEIGHBORS   = 1


class PatchCore:
    def __init__(self, coreset_ratio=CORESET_RATIO, k=K_NEIGHBORS, device='cpu'):
        self.coreset_ratio = coreset_ratio
        self.k = k
        self.device = device
        self.extractor = build_extractor(device)

        # Set after fit()
        self.memory_bank = None     # (M, C) numpy array
        self.knn = None
        self.patch_H = None
        self.patch_W = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, dataloader):
        """Build coreset memory bank from normal training images."""
        feats = extract_patchcore_features(self.extractor, dataloader, self.device)
        # feats: (N, C, H, W)
        N, C, H, W = feats.shape
        self.patch_H, self.patch_W = H, W

        # Flatten spatial positions: (N*H*W, C)
        patches = feats.permute(0, 2, 3, 1).reshape(-1, C).numpy()

        # Greedy coreset approximation via random subsampling
        n_keep = max(1, int(len(patches) * self.coreset_ratio))
        print(f'  Building memory bank: {len(patches)} -> {n_keep} patches', flush=True)
        idx = _greedy_coreset(patches, n_keep)
        self.memory_bank = patches[idx]      # (M, C)

        # Fit k-NN index
        self.knn = NearestNeighbors(n_neighbors=self.k, metric='euclidean',
                                    algorithm='auto', n_jobs=-1)
        self.knn.fit(self.memory_bank)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, dataloader, corruption_fn=None):
        """
        Returns:
          image_scores : list[float]
          pixel_maps   : list[ndarray IMG_SIZE×IMG_SIZE]
          labels       : list[int]
          gt_masks     : list[ndarray or None]
        """
        from .dataset import IMG_SIZE

        image_scores, pixel_maps, labels, gt_masks = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='  Scoring', leave=False):
                imgs, lbl, mask, _ = batch
                imgs = imgs.to(self.device)

                _, f2, f3 = self.extractor(imgs)
                H = f2.shape[2]
                f3_up = F.interpolate(f3, size=(H, H), mode='bilinear', align_corners=False)
                feats = torch.cat([f2, f3_up], dim=1)
                feats = F.avg_pool2d(feats, kernel_size=3, stride=1, padding=1)
                # feats: (B, C, H, W)

                B = imgs.shape[0]
                for b in range(B):
                    pf = feats[b].cpu().numpy()                  # (C, H, W)
                    pf = pf.transpose(1, 2, 0).reshape(-1, pf.shape[0])  # (H*W, C)

                    dists, _ = self.knn.kneighbors(pf)           # (H*W, k)
                    dist_map = dists[:, 0].reshape(H, H)         # (H, W)

                    dist_map = gaussian_filter(dist_map, sigma=4)
                    dist_map_resized = _resize_map(dist_map, IMG_SIZE)

                    image_scores.append(float(dist_map_resized.max()))
                    pixel_maps.append(dist_map_resized)
                    labels.append(int(lbl[b]))
                    gt_masks.append(mask[b].numpy() if hasattr(mask[b], 'numpy') else mask[b])

        return image_scores, pixel_maps, labels, gt_masks


# ── Coreset selection ─────────────────────────────────────────────────────────

def _greedy_coreset(patches, n_keep):
    """
    Greedy coreset subsampling (k-center / farthest-point sampling).

    To keep CPU runtime manageable we first randomly subsample a candidate pool
    of at most CANDIDATE_CAP patches, then run greedy selection on that pool.
    """
    CANDIDATE_CAP = 15_000
    n = len(patches)
    if n_keep >= n:
        return np.arange(n)

    rng = np.random.default_rng(42)
    # Candidate pool
    if n > CANDIDATE_CAP:
        pool_idx = rng.choice(n, CANDIDATE_CAP, replace=False)
        pool = patches[pool_idx]
    else:
        pool_idx = np.arange(n)
        pool = patches

    n_pool = len(pool)
    # Greedy farthest-point selection on pool
    selected_local = [int(rng.integers(0, n_pool))]
    min_dists = np.full(n_pool, np.inf)

    for _ in tqdm(range(min(n_keep - 1, n_pool - 1)),
                  desc='  Coreset selection', leave=False):
        last = pool[selected_local[-1]]
        dists_to_last = np.linalg.norm(pool - last, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        next_local = int(np.argmax(min_dists))
        selected_local.append(next_local)

    return pool_idx[np.array(selected_local)]


def _resize_map(arr, size):
    from PIL import Image
    img = Image.fromarray(arr.astype(np.float32))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)
