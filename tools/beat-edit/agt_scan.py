"""AINT GOT TIME rolls (dark FL theme, mint notes ~R71 G232 B175; OBS 1920x1200 60 fps): per native frame ->
(playhead_x or None, [note boxes x0, x1, yc], lit key rows). Grid x 92-1893, y 186-935. Playhead = teal column
(~R57 G122 B125) covering >55% of the grid height. usage: agt_scan.py clip..."""
import subprocess, json, sys, numpy as np, cv2
W, H = 1920, 1200; Y0, Y1, X0, X1 = 186, 935, 92, 1893
for clip in sys.argv[1:]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    S = []
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        F = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.int16)
        f = F[Y0:Y1, X0:X1]; B, G, R = f[..., 0], f[..., 1], f[..., 2]
        ph = (G > 95) & (B > 95) & (R < 100) & (G - R > 40) & (G < 170)
        cov = ph.mean(0); xs = np.where(cov > 0.55)[0]
        x = float(np.median(xs)) + X0 if len(xs) and xs.max() - xs.min() < 12 else None
        m = ((G > 180) & (G - R > 80) & (B > 100)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        bx = [(int(a) + X0, int(a + w) + X0, int(b + h / 2) + Y0) for a, b, w, h, ar in st[1:] if 6 <= h <= 40 and w >= 4 and ar > 20]
        k = F[Y0:Y1, 0:60]; lit = (k[..., 2] > 200) & (k[..., 2] - k[..., 0] > 90)
        rows = np.where(lit.mean(1) > 0.5)[0]
        S.append((x, bx, [int(rows.mean()) + Y0] if len(rows) else []))
    json.dump(S, open(clip + "_scan.json", "w"))
    v = [s[0] for s in S if s[0] is not None]
    print(clip, len(S), "frames; playhead in", len(v), "x", (round(min(v)), round(max(v))) if v else None, "| boxes/frame", round(np.mean([len(s[1]) for s in S]), 1), flush=True)
