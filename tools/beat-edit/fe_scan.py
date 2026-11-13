"""FEAR (green FL theme, 1920x1200 OBS): per native frame -> (None, [note boxes x0, x1, yc]) in the piano-roll grid
(x 565-1890, y 190-940). Notes are teal (~R32 G224 B144) with labels inside. usage: fe_scan.py clip"""
import subprocess, json, sys, numpy as np, cv2
clip = sys.argv[1]; W, H = 1920, 1200; Y0, Y1, X0, X1 = 190, 940, 565, 1890
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[Y0:Y1, X0:X1].astype(np.int16)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    m = ((G > 140) & (G - R > 70) & (B > 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    S.append((None, [(int(a) + X0, int(a + w) + X0, int(b + h / 2) + Y0) for a, b, w, h, ar in st[1:] if 5 <= h <= 60 and w >= 3 and ar > 15]))
json.dump(S, open(clip + "_scan.json", "w"))
print(clip, len(S), "frames | mid", S[len(S) // 2][1][:6])
