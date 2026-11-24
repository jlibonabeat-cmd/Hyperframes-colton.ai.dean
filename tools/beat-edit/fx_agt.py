"""AIN'T GOT TIME edit B + CapCut-style accents (after Jack's own CapCut pass, Nov 24): flash pulses, RGB-split glitch,
strobe into the drop, light-streak whiteout on the drop, B&W -> red-duotone flickers. Deterministic (seeded).
usage: python3 -I var/fx_agt.py in.mp4 out.mp4"""
import sys, subprocess, numpy as np, cv2
src, dst = sys.argv[1], sys.argv[2]
W, H, FPS = 1080, 1920, 30
rng = np.random.default_rng(7)
fx = {}
def add(n, *e): fx.setdefault(n, []).append(e)
for n0 in (103, 643):                                   # flash pulses: mixer cut (3.43), BASS_12 cut (21.43)
    for k, a in enumerate((0.55, 0.35, 0.18, 0.07)): add(n0 + k, "flash", a)
for k in range(3): add(257 + k, "rgb", 22 - 7 * k)     # RGB-split glitch into BASS_4 (8.57)
for n in range(386, 409):                               # strobe on the last two beats before the drop (12.86-13.63)
    if (n - 386) % 4 in (2, 3): add(n, "strobe", 0)
for k, a in ((409, 0.35), (410, 0.75)): add(k, "streak", a)            # light-streak whiteout into the drop (13.71)
for k, a in enumerate((0.95, 0.7, 0.45, 0.25, 0.1)): add(411 + k, "flash", a)
for n0 in (463, 726):                                   # B&W then red duotone: COUNTER2 (15.43), SHAPER (24.21)
    for k in range(3): add(n0 + k, "bw", 0)
    for k in range(3, 8): add(n0 + k, "duo", 0)

def flash(img, a): return cv2.addWeighted(img, 1 - a, np.full_like(img, 255), a, 0)
def rgb(img, d):
    out = img.copy(); out[..., 2] = np.roll(img[..., 2], d, 1); out[..., 0] = np.roll(img[..., 0], -d, 1)
    for _ in range(6):                                  # horizontal slice offsets
        y = int(rng.integers(0, H - 80)); h = int(rng.integers(20, 80)); s = int(rng.integers(-60, 60))
        out[y:y + h] = np.roll(out[y:y + h], s, 1)
    return out
def strobe(img):
    k = np.zeros((61, 1), np.float32); k[:] = 1 / 61
    return (cv2.filter2D(img, -1, k) * 0.45).astype(np.uint8)
def streak(img, a):
    b = img.astype(np.float32); acc = np.zeros_like(b)
    for z in np.linspace(1.0, 1.4, 9):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z); acc += cv2.warpAffine(b, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    acc /= 9; acc = acc * (1 - a) + 255 * a
    return np.clip(acc, 0, 255).astype(np.uint8)
def bw(img): g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g = cv2.convertScaleAbs(g, alpha=1.25, beta=-25); return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def duo(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255
    lo, hi = np.array([20, 10, 60], np.float32), np.array([200, 225, 255], np.float32)   # dark red -> warm white (BGR)
    return np.clip(lo + (hi - lo) * g, 0, 255).astype(np.uint8)

rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-i", src, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", "-c:a", "copy", "-shortest", dst], stdin=subprocess.PIPE)
n = 0
while True:
    buf = rd.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    img = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    for e in fx.get(n, []):
        img = {"flash": lambda i: flash(i, e[1]), "rgb": lambda i: rgb(i, e[1]), "strobe": lambda i: strobe(i),
               "streak": lambda i: streak(i, e[1]), "bw": bw, "duo": duo}[e[0]](img)
    wr.stdin.write(img.tobytes()); n += 1
wr.stdin.close(); wr.wait(); print("frames", n, "fx frames", len(fx))
