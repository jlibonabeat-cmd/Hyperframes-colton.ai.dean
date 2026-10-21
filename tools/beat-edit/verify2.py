import subprocess, sys, json, numpy as np
clip, D, a, b, inst, y0, y1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
def frame(t):
    buf = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", f"{clip}.mp4", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    return np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16)
def ph(f):
    reg = f[200:880:4]; g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(0); c = np.where(col[500:1890] > 0.6)[0]
    return c[0] + 500 if len(c) else None
h = json.load(open("hits.json"))
on = np.array(h[inst]["t"]) + 21.8182; s = np.array(h[inst]["s"])
on = on[s > 0.5]
f0 = frame(a + 0.05)
band = f0[y0:y1, 500:1890]
m = (band[..., 1] > 150) & (band[..., 1] - band[..., 2] > 60)
colm = m.any(0)
starts = np.where(np.diff(colm.astype(np.int8)) == 1)[0] + 501
print("note starts", starts.tolist())
x1, x2 = ph(frame(a + 0.5)), ph(frame(a + 1.5)); spd = (x2 - x1) if x1 and x2 else None
print("speed px/s", spd)
for t in on:
    ct = t - D
    if a < ct < b:
        x = ph(frame(ct))
        if x is None: continue
        d = starts - x
        k = np.argmin(np.abs(d))
        print(f"song {t:.3f} clip {ct:.3f} playhead {x} nearest note {starts[k]} -> {1000*d[k]/spd:+.0f} ms")
