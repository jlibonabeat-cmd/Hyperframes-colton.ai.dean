"""note pixels in the roll crop vs mp3 time, full-screen crop (x 440-865) vs split crop (x 440-1172), grid y 183-938."""
import cv2, numpy as np, json
R = {"bass": (5.75, 19.0, -1.699), "chords": (4.05, 17.3, 0.005), "counter": (7.1, 19.8, -3.052), "serum": (0.8, 14.12, 3.235)}
out = {}
for k, (a, b, D) in R.items():
    c = cv2.VideoCapture(f"{k}.mp4"); fps = c.get(5); rows = []
    for t in np.arange(a, b, 1 / 30):
        c.set(0, t * 1000); ok, f = c.read()
        if not ok: break
        g = f[205:1125, 440:1335].astype(int); Rr, G, B = g[..., 2], g[..., 1], g[..., 0]
        m = (G > 180) & (Rr > 120) & (G - Rr > 15) & (G - B > 5)
        cols = m.sum(0)
        c2 = cols >= 6
        rows.append((round(t + D, 3), int(c2[95:517].sum()), int(c2[95:].sum())))
    out[k] = rows; print(k, len(rows))
json.dump(out, open("notemap.json", "w"))
