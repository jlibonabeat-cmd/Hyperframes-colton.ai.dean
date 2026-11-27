"""J Sewell-style moody grade for Jack's face footage (requested Nov 27): exposure down, crushed blacks + S-curve,
teal shadows / warm highlights split tone, muted saturation, soft vignette. grade(img_bgr_uint8, k) with k = strength."""
import numpy as np, cv2
_cache = {}
def _vig(h, w):
    if (h, w) not in _cache:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32); d = np.sqrt(((x - w / 2) / (w * 0.62)) ** 2 + ((y - h * 0.45) / (h * 0.7)) ** 2)
        _cache[(h, w)] = np.clip(1 - 0.45 * d ** 2.0, 0.55, 1)[..., None]
    return _cache[(h, w)]
def grade(img, k=1.0):
    f = img.astype(np.float32) / 255
    f = f * (1 - 0.16 * k)                                            # v2 (Jack: a bit dark vs the screen): light exposure cut
    lum = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])[..., None]
    f = lum + (f - lum) * (1 - 0.32 * k)                              # mute saturation
    x = np.clip(f, 0, 1)                                              # S-curve, crushed toe
    f = np.clip((x - 0.02 * k) / (1 - 0.02 * k), 0, 1); f = f * f * (3 - 2 * f) * (0.55 * k) + f * (1 - 0.55 * k)
    lum = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])[..., None]
    sh = np.clip(1 - lum * 2.2, 0, 1); hi = np.clip((lum - 0.35) * 1.6, 0, 1)
    f = f + k * (sh * np.array([0.06, 0.03, -0.03]) + hi * np.array([-0.07, -0.005, 0.06]))   # BGR: teal shadows, warm highs
    f = f * (1 - k + k * _vig(*f.shape[:2]))
    return np.clip(f * 255, 0, 255).astype(np.uint8)
_kl = {}
def key_light(img, cx, cy, sx=150, sy=230, amt=0.22):
    """soft warm key on one side of the face: a gaussian pool brightening mids/highs (screen-ish), slightly warm"""
    h, w = img.shape[:2]; key = (h, w, cx, cy, sx, sy)
    if key not in _kl:
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        _kl[key] = np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2) / 2)[..., None]
    m = _kl[key] * amt; f = img.astype(np.float32) / 255
    lit = 1 - (1 - f) * (1 - f * np.array([0.85, 0.95, 1.05]))       # screen with itself, a touch warm (BGR)
    return np.clip((f * (1 - m) + lit * m) * 255, 0, 255).astype(np.uint8)
