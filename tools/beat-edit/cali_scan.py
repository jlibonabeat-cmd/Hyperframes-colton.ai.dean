"""Decode a piano-roll recording at native 60 fps; per frame: playhead x (full-height bright column) + note boxes
(playhead masked +-2 px only). Saves <clip>_scan.json."""
import subprocess, json, sys, numpy as np, cv2
def notemask(f):
    return (f[..., 1] > 170) & (f[..., 1] - f[..., 2] > 80) & (f[..., 0] > 100)
def ph(f):
    reg = notemask(f[200:870:5]); c = reg.mean(0); xs = np.where(c[352:] > 0.85)[0]
    return int(xs[-1]) + 352 if len(xs) else None
def boxes(f, x):
    m = notemask(f).astype(np.uint8)
    m[:185] = 0; m[878:] = 0; m[:, :352] = 0
    if x is not None: m[:, max(0, x - 2):x + 3] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x0), int(y0 + h / 2), int(w)) for x0, y0, w, h, a in st[1:] if a > 40 and h >= 8]
clip = sys.argv[1]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(1080 * 1920 * 3)
    if len(buf) < 1080 * 1920 * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); x = ph(f); S.append((x, boxes(f, x)))
json.dump(S, open(clip + "_scan.json", "w")); print(clip, len(S), "frames; playhead found in", sum(s[0] is not None for s in S))
