"""AGT new-layout rolls (blue-grey theme, browser open; notes pale green ~R151 G199 B161): per frame ->
(playhead_x from the ruler marker triangle y 158-174, [note boxes x0, x1, yc]). Grid x 520-1893, y 183-938."""
import subprocess, json, sys, numpy as np, cv2
W, H = 1920, 1200; Y0, Y1, X0, X1 = 183, 938, 520, 1893
for clip in sys.argv[1:]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    S = []
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        F = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.int16)
        r = F[158:174, X0 - 10:X1]; mk = ((r[..., 1] > 170) & (r[..., 0] < 140)).any(0); xs = np.where(mk)[0]
        x = float((xs.min() + xs.max()) / 2) + X0 - 10 if len(xs) and 5 <= xs.max() - xs.min() <= 26 else None
        f = F[Y0:Y1, X0:X1]; B, G, R = f[..., 0], f[..., 1], f[..., 2]
        m = ((G > 180) & (R > 120) & (G - R > 15) & (G - B > 5)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        bx = [(int(a) + X0, int(a + w) + X0, int(b + h / 2) + Y0) for a, b, w, h, ar in st[1:] if 6 <= h <= 40 and w >= 4 and ar > 20]
        S.append((x, bx, []))
    json.dump(S, open(clip + "_scan.json", "w"))
    v = [s[0] for s in S if s[0] is not None]
    print(clip, len(S), "frames; playhead in", len(v), "x", (round(min(v)), round(max(v))) if v else None, "| boxes/frame", round(np.mean([len(s[1]) for s in S]), 1), flush=True)
