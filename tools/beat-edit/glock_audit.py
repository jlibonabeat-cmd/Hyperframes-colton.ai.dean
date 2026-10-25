"""Visual note-hit times per piano-roll recording (clip time), then per-bar precision/recall vs the master's onsets
for shifts D, D+8 bars, D+16 bars."""
import subprocess, json, sys, numpy as np, cv2
from scipy.signal import butter, sosfilt, find_peaks
BAR = 240 / 149; B0 = 0.032; SR = 22050
V = {"snare": ("snare.mp4", 11.376, 1.4, 8.6, (1500, 5000)), "hat": ("hat.mp4", 11.719, 1.0, 4.2, (7000, 10500)),
     "kick": ("kick.mp4", 11.83, 1.0, 13.8, (35, 120)), "808a": ("s808a.mp4", 10.131, 5.6, 10.6, (30, 90)),
     "808b": ("s808b.mp4", 8.951, 3.8, 16.4, (30, 90)), "chops": ("chops.mp4", 11.688, 1.0, 14.6, (250, 2500)),
     "scratch": ("scratch.mp4", 9.757, 3.0, 8.6, (300, 6000))}
def ph(f):
    reg = f[200:880:4]; g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    c = np.where(g.mean(0)[300:1890] > 0.6)[0]; return c[0] + 300 if len(c) else None
def notes(f, x=None):
    m = ((f[..., 1] > 170) & (f[..., 1] - f[..., 2] > 70)).astype(np.uint8)[150:880, 300:1895]
    if x is not None:
        m[:, max(0, x - 300 - 35):max(0, x - 300 + 35)] = 0          # the playhead and its glow are green too
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    res = []
    for x, y, w, h, a in st[1:]:
        if not (8 <= h <= 36 and w >= 4 and a > 30): continue
        ym = y + h // 2; row = f[150 + ym - 1:150 + ym + 2, 300 + x:300 + x + w]
        on = ((row[..., 1] > 190) & (row[..., 1] - row[..., 2] > 90)).all(0)
        d = np.diff(np.concatenate([[0], on.astype(np.int8)]))
        for xs in np.where(d == 1)[0]: res.append(int(x + xs) + 300)
    return sorted(set(res))
def master_onsets(lo, hi):
    y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); e = np.convolve(e, np.ones(110) / 110, mode="same")[::110]
    le = np.log1p(e * 300); fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 93), distance=12)
    return pk * 0.005, fl[pk] / fl[pk].max()
out = {}
for name in (sys.argv[1:] or V):
    clip, D, a, b, band = V[name]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", clip, "-vf", "fps=60", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    k = 0; vis = []; phx = []; hist = []
    while True:
        buf = p.stdout.read(1080 * 1920 * 3)
        if len(buf) < 1080 * 1920 * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); t = a + k / 60; x = ph(f)
        phx.append((round(t, 3), x))
        hist.append((x, notes(f, x)))
        # notes seen 25 frames ago (playhead well away then), crossed by the playhead between the last frame and this one
        if x is not None and len(hist) > 26 and hist[-2][0] is not None and 0 < x - hist[-2][0] < 40:
            x0 = hist[-2][0]
            for n_ in hist[-26][1]:
                if x0 < n_ <= x: vis.append(t - 1 / 60 + (n_ - x0) / (x - x0) / 60)
        k += 1
    mo, ms = master_onsets(*band); strong = mo[ms > 0.35]
    vis = np.array(sorted(vis))
    print(f"{name}: {len(vis)} visual hits; playhead present {sum(1 for _, x in phx if x is not None)}/{len(phx)} frames")
    res = {}
    for sh in (0, 8, 16):
        Dp = D + sh * BAR; row = []
        for bi in range(4, 24):
            s0 = B0 + bi * BAR
            if s0 - Dp < a or s0 + BAR - Dp > b: row.append("   .    "); continue
            vv = vis[(vis + Dp >= s0) & (vis + Dp < s0 + BAR)] + Dp; ss = strong[(strong >= s0) & (strong < s0 + BAR)]
            prec = sum(np.min(np.abs(mo - x)) < 0.045 for x in vv); rec = sum(np.min(np.abs(vis + Dp - x)) < 0.045 for x in ss) if len(vis) else 0
            row.append(f"{prec}/{len(vv)} {rec}/{len(ss)}".rjust(8)); res[f"{sh}:{bi}"] = (int(prec), len(vv), int(rec), len(ss))
        print(f"   +{sh:2d} bars", " ".join(row))
    out[name] = {"vis": vis.tolist(), "res": res}
print("bars       ", " ".join(f"{b:8d}" for b in range(4, 24)))
json.dump(out, open("audit.json", "w"))
