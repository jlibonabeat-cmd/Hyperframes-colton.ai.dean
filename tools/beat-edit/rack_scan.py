"""Rack On Top piano rolls (dark theme, mint or salmon notes): per native frame (playhead_x, note boxes). <clip>_scan.json
Playhead = full-height green column; notes = bright saturated boxes."""
import subprocess, json, sys, numpy as np, cv2
def notemask(f):
    mx = f.max(axis=2); mn = f.min(axis=2)
    return (mx > 165) & (mx - mn > 60)
def ph(f):
    reg = f[220:880:4].astype(np.int16)
    g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5) & (reg[..., 1] > 90)
    xs = np.where(g.mean(0)[457:1890] > 0.8)[0]
    return int(xs[0]) + 457 if len(xs) else None
def boxes(f, x):
    m = notemask(f).astype(np.uint8)
    m[:215] = 0; m[892:] = 0; m[:, :457] = 0
    if x is not None: m[:, max(0, x - 3):x + 4] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x0), int(y0 + h / 2), int(w)) for x0, y0, w, h, a in st[1:] if a > 60 and h >= 10 and h <= 40]
clip = sys.argv[1]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(1080 * 1920 * 3)
    if len(buf) < 1080 * 1920 * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); x = ph(f); S.append((x, boxes(f, x)))
json.dump(S, open(clip + "_scan.json", "w")); print(clip, len(S), "frames; playhead in", sum(s[0] is not None for s in S),
  "median x", int(np.median([s[0] for s in S if s[0] is not None])) if any(s[0] for s in S) else None)
