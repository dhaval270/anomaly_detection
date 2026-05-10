import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224


def get_transforms(augment_corruption=False, corruption_fn=None):
    """Return image transform pipeline."""
    base = [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(base)


class MVTecDataset(Dataset):

    def __init__(self, root, category, split='train', transform=None,
                 corruption_fn=None):
        self.root = os.path.join(root, category)
        self.split = split
        self.transform = transform
        self.corruption_fn = corruption_fn  # applied to PIL image before transform
        self.samples = []
        self.masks = []
        self.labels = []   # 0 = good, 1 = anomaly
        self.defect_types = []

        if split == 'train':
            good_dir = os.path.join(self.root, 'train', 'good')
            for fname in sorted(os.listdir(good_dir)):
                if fname.endswith('.png'):
                    self.samples.append(os.path.join(good_dir, fname))
                    self.masks.append(None)
                    self.labels.append(0)
                    self.defect_types.append('good')
        else:
            test_dir = os.path.join(self.root, 'test')
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                label = 0 if defect_type == 'good' else 1
                mask_dir = None
                if label == 1:
                    mask_dir = os.path.join(self.root, 'ground_truth', defect_type)
                for fname in sorted(os.listdir(defect_dir)):
                    if fname.endswith('.png'):
                        self.samples.append(os.path.join(defect_dir, fname))
                        self.labels.append(label)
                        self.defect_types.append(defect_type)
                        if mask_dir:
                            mname = fname.replace('.png', '_mask.png')
                            mpath = os.path.join(mask_dir, mname)
                            self.masks.append(mpath if os.path.exists(mpath) else None)
                        else:
                            self.masks.append(None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.corruption_fn is not None:
            img = self.corruption_fn(img)
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])(img)

        # Load pixel-level mask; return zeros if not available (normal images)
        if self.masks[idx] is not None:
            m = Image.open(self.masks[idx]).convert('L')
            m = m.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            mask = (np.array(m) > 0).astype(np.uint8)
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        return tensor, self.labels[idx], mask, self.samples[idx]
