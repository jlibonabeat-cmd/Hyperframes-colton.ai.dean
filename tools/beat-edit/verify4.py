"""Every note the playhead crosses in a song window -> song time; nearest master onset (any strength) of inst."""
import subprocess, sys, json, numpy as np, cv2
clip, D, s0, s1, inst = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
PH = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None
p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{s0 - D:.3f}", "-t", f"{s1 - s0:.3f}", "-i", f"{clip}.mp4", "-vf", "fps=60",
                      "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
def ph(f):
    if PH: return PH
    reg = f[200:880:4]; g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(0); c = np.where(col[500:1890] > 0.6)[0]
    return c[0] + 500 if len(c) else None
def notes(f):
    m = ((f[..., 1] > 170) & (f[..., 1] - f[..., 2] > 70)).astype(np.uint8)[130:930, 500:1895]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(x + 500, y + 130, w, h) for x, y, w, h, a in st[1:] if 8 <= h <= 40 and w >= 6 and a > 40]
prev = None; k = 0; cross = []
while True:
    b = p.stdout.read(1080 * 1920 * 3)
    if len(b) < 1080 * 1920 * 3: break
    f = np.frombuffer(b, np.uint8).reshape(1080, 1920, 3).astype(np.int16)
    x = ph(f); t = s0 + k / 60
    if PH:   # follow mode: notes move left past fixed x
        ns = notes(f)
        if prev is not None:
            for nx, ny, w, hh in prev:
                if nx >= PH and any(abs(ny - my) < 4 and nx - 30 <= mx < PH and mx >= nx - 30 for mx, my, _, _ in ns if mx < PH and mx > nx - 30):
                    cross.append(t)
        prev = ns
    elif x is not None:
        if prev is not None and prev[0] is not None and 0 < x - prev[0] < 20:
            for nx, ny, w, hh in prev[1]:
                if prev[0] < nx <= x:
                    cross.append(t - (x - nx) / (x - prev[0]) / 60)
        prev = (x, notes(f))
    k += 1
h = json.load(open("hits.json"))
on = np.array(h[inst]["t"]) + 21.8182; s = np.array(h[inst]["s"])
cross = sorted(set(round(c, 3) for c in cross))
print("notes crossed:", len(cross))
for c in cross:
    d = on - c; i = np.argmin(np.abs(d))
    print(f"  note at song {c:.3f}  nearest {inst} onset {on[i]:.3f} (strength {s[i]:.2f})  diff {1000*d[i]:+.0f} ms")
oo = [(round(o, 3), round(float(ss), 2)) for o, ss in zip(on, s) if s0 < o < s1 and ss > 0.3]
print("master onsets in window:", oo)
