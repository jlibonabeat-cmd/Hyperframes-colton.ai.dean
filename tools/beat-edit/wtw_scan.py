"""Beige FL theme piano rolls: per native frame (playhead_x, note boxes). Playhead = red column (R-G>50); notes = brown
boxes (R 115-175, R-B 30-60, max < 190)."""
import subprocess, json, sys, numpy as np, cv2
def notemask(f):
    R, G, B = f[..., 2], f[..., 1], f[..., 0]
    return (((R > 110) & (R < 180) & (R - B > 30) & (R - B < 65) & (G < 160)) | ((np.maximum(np.maximum(R, G), B) < 150) & (R - B > 25) & (R > 45)))
def ph(f):
    reg = f[260:940:4]; g = (reg[..., 2] - reg[..., 1] > 45) & (reg[..., 2] > 200)
    xs = np.where(g.mean(0)[455:1890] > 0.6)[0]
    return int(xs[len(xs) // 2]) + 455 if len(xs) else None
def boxes(f, x):
    m = notemask(f).astype(np.uint8)
    m[:255] = 0; m[935:] = 0; m[:, :455] = 0
    if x is not None: m[:, max(0, x - 8):x + 9] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x0), int(y0 + h / 2), int(w)) for x0, y0, w, h, a in st[1:] if a > 40 and 12 <= h <= 70]
clip = sys.argv[1]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(1080 * 1920 * 3)
    if len(buf) < 1080 * 1920 * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); x = ph(f); S.append((x, boxes(f, x)))
json.dump(S, open(clip + "_scan.json", "w")); print(clip, len(S), "frames; playhead in", sum(s[0] is not None for s in S), "boxes mid", S[len(S) // 2][1][:6])
