"""SCOUSE (blue-grey FL theme, OBS 1920x1200): per native frame -> (playhead_x or None, [note boxes x0, x1, yc]).
Notes pale mint (~R165 G215 B176), FL playhead a green line (~R80 G121 B64). Grid x 356-1895, y 185-925.
usage: sc_scan.py clip"""
import subprocess, json, sys, numpy as np, cv2
clip = sys.argv[1]; W, H = 1920, 1200; Y0, Y1, X0, X1 = 185, 925, 356, 1895
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[Y0:Y1, X0:X1].astype(np.int16)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    ph = (G > 105) & (G - R > 25) & (G - B > 35) & (R < 120)
    cov = ph[::3].mean(0); xs = np.where(cov > 0.5)[0]
    x = float(np.median(xs)) + X0 if len(xs) and xs.max() - xs.min() < 12 else None
    m = ((G > 180) & (R > 125) & (B > 125) & (G - R > 15) & (G - B > 8)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    S.append((x, [(int(a) + X0, int(a + w) + X0, int(b + h / 2) + Y0) for a, b, w, h, ar in st[1:] if 4 <= h <= 45 and w >= 3 and ar > 12]))
json.dump(S, open(clip + "_scan.json", "w"))
v = [s[0] for s in S if s[0] is not None]
print(clip, len(S), "frames; playhead in", len(v), "x range", (min(v), max(v)) if v else None, "| boxes/frame", round(np.mean([len(s[1]) for s in S]), 1))
