"""For every piano-roll view: song times at which a note start reaches the play position (what the viewer SEES hit),
compared with what is HEARD (master onsets of that instrument). Writes audit.json."""
import subprocess, json, numpy as np, cv2
VIEWS = {  # name: clip, D, clip_a, clip_b, inst, fixed play x (None = moving playhead)
    "HAT": ("hat.mp4", 16.561, 5.2, 11.5, "hat", 909),
    "OPENHAT": ("openhat.mp4", 16.039, 5.9, 11.5, "hat", None),
    "SNARE_A": ("snare_a.mp4", 20.854, 1.1, 7.9, "snare", None),
    "SNARE_M": ("snare_main.mp4", 12.35, 9.5, 19.3, "snare", None),
    "808A": ("s808.mp4", 28.93, 4.9, 14.5, "kick", 550),
    "808B": ("s808b.mp4", 35.234, 3.1, 8.5, "kick", None),
}
SPEED_FOLLOW = 406 / (240 / 176)          # px/s the notes scroll in the follow-mode hat roll
def ph(f):
    reg = f[200:880:4]; g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(0); c = np.where(col[500:1890] > 0.6)[0]
    return c[0] + 500 if len(c) else None
def notes(f):
    """note starts: in each note band, columns where a strongly-mint run begins on the band's middle rows."""
    m = ((f[..., 1] > 170) & (f[..., 1] - f[..., 2] > 70)).astype(np.uint8)[130:930, 500:1895]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    res = []
    for x, y, w, h, a in st[1:]:
        if not (14 <= h <= 40 and w >= 6 and a > 80):
            continue
        ym = y + h // 2
        row = f[130 + ym - 2:130 + ym + 3, 500 + x:500 + x + w]
        on = ((row[..., 1] > 200) & (row[..., 1] - row[..., 2] > 100)).all(0)
        d = np.diff(np.concatenate([[0], on.astype(np.int8)]))
        for xs in np.where(d == 1)[0]:
            res.append((int(x + xs) + 500, int(y) + 130, 10, int(h)))
    return res


h = json.load(open("hits.json"))
out = {}
for name, (clip, D, a, b, inst, PX) in VIEWS.items():
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", clip, "-vf", "fps=60",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    prev = None; k = 0; cross = []; phs = []; spd = None; first = None
    while True:
        buf = p.stdout.read(1080 * 1920 * 3)
        if len(buf) < 1080 * 1920 * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16)
        t = a + k / 60 + D
        ns = notes(f)
        if PX is not None:
            if spd is None and k == 6:
                g0 = cv2.cvtColor(first.astype(np.uint8), cv2.COLOR_BGR2GRAY)[150:930, 600:1880].astype(np.float32)
                g1 = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY)[150:930, 600:1880].astype(np.float32)
                (dx, dy), _ = cv2.phaseCorrelate(g0, g1)
                spd = -dx * 10.0                              # px/s leftwards (6 frames = 0.1 s)
                print(name, "scroll speed px/s", round(spd, 1))
            if k == 0:
                first = f
            if spd:
                for nx, ny, w, hh in ns:
                    if PX <= nx < PX + spd / 60:
                        cross.append(t + (nx - PX) / spd)
        else:
            x = ph(f)
            if x is not None and prev is not None and prev[0] is not None and 0 < x - prev[0] < 20:
                for nx, ny, w, hh in prev[1]:
                    if prev[0] < nx <= x:
                        cross.append(t - 1 / 60 + (nx - prev[0]) / (x - prev[0]) / 60)
            prev = (x, ns)
            phs.append((round(t, 3), x))
        k += 1
    on = np.array(h[inst]["t"]) + 21.8182; s = np.array(h[inst]["s"])
    cr = sorted(cross)
    merged = []
    for c in cr:
        if not merged or c - merged[-1] > 0.12: merged.append(c)
    errs = []
    for c in merged:
        if 21.82 < c < 49.09:
            d = on - c; i = np.argmin(np.abs(d)); errs.append((round(c, 3), round(1000 * float(d[i])), round(float(s[i]), 2)))
    out[name] = {"song_range": [a + D, b + D], "hits": errs}
    e = np.array([abs(x[1]) for x in errs]) if errs else np.array([999])
    print(f"{name:8s} song {a + D:6.2f}-{b + D:6.2f}  visible hits {len(errs):3d}  median |err| {np.median(e):4.0f} ms  >60ms: {int((e > 60).sum())}")
json.dump(out, open("audit.json", "w"), indent=0)
