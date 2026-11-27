"""Jack's CapCut transition looks (Mystery v3 -> his edit, Nov 27) rebuilt as deterministic overlays on a rendered edit.
Types: warm light-leak burn, prism/RGB leak + chromatic glitch, rainbow prism bands + glow + white flash,
red film-burn / ink wipe, white flash, white light streak. Like his, they sit in the top ~half of the frame and fade
out before the lower note area. Audio copied. usage: fx_overlays.py in.mp4 out.mp4 [events.json]"""
import sys, json, subprocess, numpy as np, cv2

SRC, OUT = sys.argv[1:3]
W, H = 1080, 1920
LW, LH = 270, 480            # layers drawn at quarter res, then upscaled (they are all soft)

# frame -> list of (effect, phase 0..1, strength); defaults = the frames Jack used on Mystery v3 (same timeline)
EVENTS = [("warm", 41, 46), ("prism", 185, 201), ("rainbow", 325, 341), ("red", 465, 482), ("flash", 495, 496),
          ("redring", 497, 504), ("streak", 504, 505)]
if len(sys.argv) > 3:
    EVENTS = [tuple(e) for e in json.load(open(sys.argv[3]))]

yy, xx = np.mgrid[0:LH, 0:LW].astype(np.float32)
xx /= LW; yy /= LH
# vertical mask: full over the top half, gone by ~62% (keeps the lower notes readable, like his overlays)
VMASK = np.clip((0.62 - yy) / 0.14, 0, 1)[..., None]


def noise(seed, cells=6):
    r = np.random.default_rng(seed).random((cells, int(cells * 1.8))).astype(np.float32)
    return cv2.resize(r, (LW, LH), interpolation=cv2.INTER_CUBIC)


def up(layer):
    if layer.ndim == 3 and layer.shape[-1] == 1:
        layer = np.repeat(layer, 3, -1)
    return cv2.resize(layer, (W, H), interpolation=cv2.INTER_LINEAR)


def screen(img, layer, a):
    return 1 - (1 - img) * (1 - np.clip(layer * a, 0, 1))


def hue_bgr(h):
    """h in 0..1 -> BGR in 0..1 (full saturation)."""
    hsv = np.stack([h * 179, np.full_like(h, 255), np.full_like(h, 255)], -1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255


def chroma(img, px):
    out = img.copy()
    out[..., 2] = np.roll(img[..., 2], px, 1)
    out[..., 0] = np.roll(img[..., 0], -px, 1)
    return out


def bloom(img, a):
    b = cv2.GaussianBlur(np.clip(img - 0.55, 0, 1), (0, 0), 18)
    return np.clip(img + b * a, 0, 1)


def warm(img, u, n):
    env = [0.55, 1.0, 0.92, 0.7, 0.3][min(4, int(u * 5))]
    cx, cy = 0.45 + 0.12 * u, 0.22
    d = np.sqrt(((xx - cx) / 0.85) ** 2 + ((yy - cy) / 0.5) ** 2)
    tex = 0.75 + 0.25 * noise(11 + n, 7)
    core = np.exp(-d ** 2 * 1.6) * tex
    col = np.dstack([0.35 + 0.1 * core, 0.75 + 0.2 * core, np.ones_like(core)])          # yellow
    edge = np.exp(-((xx - (0.92 - 0.05 * u)) / 0.05) ** 2) * np.exp(-((yy - 0.25) / 0.3) ** 2)  # red-orange burnt rim
    layer = col * np.clip(core * 1.6, 0, 1)[..., None] + np.dstack([0.1 * edge, 0.25 * edge, 0.9 * edge])
    L = up(layer * VMASK)
    return screen(img, L, env)


def prism(img, u, n):
    rs = np.random.default_rng(300 + n)
    flick = [0.6, 0.9, 1.0, 0.5, 0.25, 0.8, 0.3, 0.1, 0.0, 0.55, 0.85, 0.4, 0.15, 0.0, 0.3, 0.6][n % 16]
    ang = 0.6 + 0.3 * np.sin(n * 0.9)
    s = xx * np.cos(ang) + yy * np.sin(ang)                                                   # diagonal streaks
    streak = np.clip(0.5 + 0.5 * np.sin(s * 14 + n * 1.7) * np.sin(s * 5 - n * 0.6), 0, 1) ** 1.5
    corner = np.exp(-(((xx - 0.85) / 0.35) ** 2 + ((yy - (0.08 + 0.35 * u)) / 0.22) ** 2))
    corner += 0.7 * np.exp(-(((xx - 0.2 - 0.6 * u) / 0.3) ** 2 + ((yy - 0.4) / 0.15) ** 2))
    hue = (s * 1.6 + n * 0.07) % 1.0
    col = hue_bgr(hue) * 0.45 + 0.55                                                           # washed-out rainbow, hot core
    grain = (rs.random((LH, LW)) > 0.985).astype(np.float32) * corner                        # dust sparkles
    layer = col * (streak * corner)[..., None] + grain[..., None]
    out = screen(img, up(layer * VMASK), flick)
    px = int(round(10 * flick))
    if px:
        out = chroma(out, px)
    if flick > 0.7:                                                                           # one sliced band jump
        y0 = int(rs.uniform(0.1, 0.45) * H); hgt = int(rs.uniform(40, 110))
        out[y0:y0 + hgt] = np.roll(out[y0:y0 + hgt], int(rs.uniform(-40, 40)), 1)
    return out


def rainbow(img, u, n):
    if n >= 14:                                                                               # final 2 frames: glow + white flash
        out = bloom(img, 1.6)
        top = np.clip((0.5 - yy) / 0.25, 0, 1)[..., None] * (0.85 if n == 15 else 0.45)
        return screen(out, up(np.ones_like(VMASK) * top), 1.0)
    if n >= 10:                                                                               # warm burnt flare
        d = np.sqrt(((xx - 0.6) / 0.5) ** 2 + ((yy - 0.3) / 0.4) ** 2)
        lay = np.dstack([0.15 + 0 * d, 0.55 + 0 * d, 0.95 + 0 * d]) * np.exp(-d ** 2)[..., None]
        return screen(img, up(lay * VMASK), 0.8)
    pos = (xx - (0.05 + 0.09 * n)) * 3.2                                                      # vertical prism bands drifting right
    band = np.exp(-(pos % 1.0 - 0.5) ** 2 / 0.05)
    hue = (0.72 + 0.35 * pos) % 1.0                                                           # purple -> blue -> orange
    col = hue_bgr(hue) * 0.55 + 0.45
    wash = 0.35 + 0.25 * np.sin(n * 1.3) ** 2
    layer = col * band[..., None] + wash
    return chroma(screen(img, up(layer * VMASK), 0.85), 4)


def red_tint(img, a, mask):
    lum = img.mean(-1, keepdims=True)
    red = np.dstack([lum * 0.15, lum * 0.12, 0.35 + lum * 0.9])
    m = up(mask)[..., None] * a
    return img * (1 - m) + red * m


def red(img, u, n):
    if n < 8:                                                                                 # burn creeping in from the top-right
        r = 0.25 + 0.12 * n
        d = np.sqrt(((xx - 1.05) / 1.0) ** 2 + ((yy + 0.05) / 0.55) ** 2)
        m = np.clip((r - d) / 0.08, 0, 1) * VMASK[..., 0]
        rim = np.exp(-((d - r) / 0.03) ** 2) * VMASK[..., 0]
        out = red_tint(img, 0.75, m)
        return screen(out, up(np.dstack([0.05 * rim, 0.15 * rim, 0.9 * rim])), 0.9)
    k = n - 8                                                                                 # red flood with ink blobs, fading
    a = [1.0, 1.0, 0.95, 0.9, 0.85, 0.7, 0.45, 0.2, 0.05][min(k, 8)]
    out = red_tint(img, a, VMASK[..., 0])
    if k < 3:
        blobs = (noise(700 + k, 5) > 0.72).astype(np.float32)
        blobs = cv2.GaussianBlur(blobs, (0, 0), 3) * VMASK[..., 0]
        out = out * (1 - 0.85 * up(blobs)[..., None])
    return out


def flash(img, u, n):
    return screen(img, up(np.ones((LH, LW, 3), np.float32) * VMASK), 0.95)


def redring(img, u, n):
    a = [0.6, 0.95, 1.0, 0.9, 0.6, 0.3, 0.1][min(n, 6)]
    cx = 0.75 - 0.25 * u; cy = 0.28; r = 0.32 + 0.1 * u
    d = np.sqrt(((xx - cx) / 0.9) ** 2 + ((yy - cy) / 0.5) ** 2)
    inside = np.clip((r - d) / 0.03, 0, 1) * VMASK[..., 0]
    outside = np.clip((d - r) / 0.03, 0, 1) * VMASK[..., 0]
    out = img * (1 - 0.8 * a * up(inside)[..., None])                                          # dark disc
    out = red_tint(out, 0.85 * a, outside)                                                    # red outside the ring
    return out


def streak(img, u, n):
    col = np.exp(-((xx - 0.32) / 0.07) ** 2) * VMASK[..., 0]
    return screen(img, up(np.repeat(col[..., None], 3, -1)), 1.0)


FX = dict(warm=warm, prism=prism, rainbow=rainbow, red=red, flash=flash, redring=redring, streak=streak)

rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", SRC,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", "-c:a", "copy", "-shortest", OUT], stdin=subprocess.PIPE)
fn = 0; hit = 0
while True:
    b = rd.stdout.read(W * H * 3)
    if len(b) < W * H * 3:
        break
    img = np.frombuffer(b, np.uint8).reshape(H, W, 3)
    for name, a_, b_ in EVENTS:
        if a_ <= fn < b_:
            f = img.astype(np.float32) / 255
            f = FX[name](f, (fn - a_) / max(1, b_ - a_ - 1), fn - a_)
            img = (np.clip(f, 0, 1) * 255).astype(np.uint8); hit += 1
    wr.stdin.write(np.ascontiguousarray(img).tobytes()); fn += 1
wr.stdin.close(); wr.wait(); print("frames", fn, "fx frames", hit)
