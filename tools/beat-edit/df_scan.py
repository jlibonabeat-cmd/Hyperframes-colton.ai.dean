"""DRUG FIT (light FL theme, pink notes; OBS 1920x1200 60 fps): per native frame -> (playhead_x or None, [note boxes
x0, x1, yc], [lit key rows]). Grid x 354-1893, y 180-937. Notes hot pink (~R245 G71 B163). FL's playhead = a soft pink
band whose sharp right edge is the play position; its marker triangle in the ruler (y 155-175) is centred on it, so the
marker gives x. Lit keys orange (~R213 G154 B95) in the keyboard x 262-352. usage: df_scan.py clip..."""
import subprocess, json, sys, numpy as np, cv2
W, H = 1920, 1200; Y0, Y1, X0, X1 = 180, 937, 354, 1893
for clip in sys.argv[1:]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    S = []
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        F = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.int16)
        r = F[157:173, X0:X1]; mk = ((r[..., 2] > 200) & (r[..., 2] - r[..., 1] > 90)).any(0)
        xs = np.where(mk)[0]
        x = float((xs.min() + xs.max()) / 2) + X0 if len(xs) and 6 <= xs.max() - xs.min() <= 24 else None
        f = F[Y0:Y1, X0:X1]; B, G, R = f[..., 0], f[..., 1], f[..., 2]
        m = ((R > 200) & (R - G > 120) & (B > 110) & (B < 210)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        bx = [(int(a) + X0, int(a + w) + X0, int(b + h / 2) + Y0) for a, b, w, h, ar in st[1:] if 4 <= h <= 45 and w >= 3 and ar > 12]
        k = F[Y0:Y1, 266:350]; lit = (k[..., 2] > 185) & (k[..., 2] - k[..., 0] > 80) & (k[..., 1] > 120) & (k[..., 1] < 185)
        rows = np.where(lit.mean(1) > 0.5)[0]
        S.append((x, bx, [int(rows.mean()) + Y0] if len(rows) else []))
    json.dump(S, open(clip + "_scan.json", "w"))
    v = [s[0] for s in S if s[0] is not None]
    print(clip, len(S), "frames; marker in", len(v), "x", (round(min(v)), round(max(v))) if v else None, "| boxes/frame", round(np.mean([len(s[1]) for s in S]), 1), "| lit frames", sum(bool(s[2]) for s in S), flush=True)
