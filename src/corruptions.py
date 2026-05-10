"""Image corruption functions for robustness evaluation.

Each corruption takes a PIL Image and a severity level (1-5) and returns a PIL Image.
"""
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2


# ── Severity parameters ───────────────────────────────────────────────────────

GAUSSIAN_NOISE_SIGMAS = [0.04, 0.08, 0.12, 0.18, 0.26]   # fraction of 255
GAUSSIAN_BLUR_SIGMAS  = [0.5,  1.0,  2.0,  3.0,  4.5]
BRIGHTNESS_FACTORS    = [1.5,  1.9,  2.4,  2.9,  3.5]     # multiplicative
JPEG_QUALITIES        = [80,   60,   40,   25,   10]


# ── Individual corruptions ────────────────────────────────────────────────────

def gaussian_noise(img: Image.Image, severity: int = 1) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    sigma = GAUSSIAN_NOISE_SIGMAS[severity - 1] * 255.0
    noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def gaussian_blur(img: Image.Image, severity: int = 1) -> Image.Image:
    sigma = GAUSSIAN_BLUR_SIGMAS[severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def brightness_change(img: Image.Image, severity: int = 1) -> Image.Image:
    factor = BRIGHTNESS_FACTORS[severity - 1]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def jpeg_compression(img: Image.Image, severity: int = 1) -> Image.Image:
    quality = JPEG_QUALITIES[severity - 1]
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


CORRUPTIONS = {
    'gaussian_noise':    gaussian_noise,
    'gaussian_blur':     gaussian_blur,
    'brightness_change': brightness_change,
    'jpeg_compression':  jpeg_compression,
}

CORRUPTION_LABELS = {
    'gaussian_noise':    'Gaussian Noise',
    'gaussian_blur':     'Gaussian Blur',
    'brightness_change': 'Brightness Change',
    'jpeg_compression':  'JPEG Compression',
}

N_SEVERITIES = 5


def get_corruption_fn(name: str, severity: int):
    """Return a single-argument callable: PIL -> PIL."""
    fn = CORRUPTIONS[name]
    return lambda img: fn(img, severity)


def apply_random_corruption(img: Image.Image, severity_range=(1, 2)) -> Image.Image:
    """Randomly apply one of the four corruptions at a random mild severity."""
    name = np.random.choice(list(CORRUPTIONS.keys()))
    severity = int(np.random.randint(severity_range[0], severity_range[1] + 1))
    return CORRUPTIONS[name](img, severity)
