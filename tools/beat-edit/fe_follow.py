"""Follow-mode note hits (MERSEY): the roll scrolls left under a fixed (possibly hidden) playhead at x = Xp. For each
note track, find when its left edge crosses Xp. If Xp isn't given, search it so the crossings best match master onsets
in the band. usage: mer_follow.py clip band [Xp]  -> {clip}_hits.json, prints offsets"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band = sys.argv[1], sys.argv[2]; XP = float(sys.argv[3]) if len(sys.argv) > 3 else None
FPS = 60; BAR = 240 / 110; OF = 0.02
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
def Dp(t):
    for a, b, D, *_ in C:
        if a - 0.05 <= t <= b + 0.05: return D
    return None
# tracks of left edges (notes move left a few px per frame)
tracks, open_ = [], []
for i, (p, bx) in enumerate(S):
    bx = [b for b in bx if b[2] > 195]; used = set(); nxt = []
    for tr in open_:
        f, x, y = tr[-1]
        if i - f > 4: tracks.append(tr); continue
        v = (tr[-1][1] - tr[-3][1]) / (tr[-1][0] - tr[-3][0]) if len(tr) >= 3 else -8
        pred = x + v * (i - f)
        c = [k for k, (a, b, yy) in enumerate(bx) if k not in used and abs(yy - y) <= 5 and abs(a - pred) <= 14]
        if c:
            k = min(c, key=lambda k: abs(bx[k][0] - pred)); used.add(k); tr.append((i, bx[k][0], bx[k][2]))
        nxt.append(tr)
    for k, (a, b, yy) in enumerate(bx):
        if k not in used: nxt.append([(i, a, yy)])
    open_ = nxt
tracks += open_
tracks = [np.array(t, float) for t in tracks if len(t) >= 10]
v = np.median([np.polyfit(t[:, 0], t[:, 1], 1)[0] for t in tracks])
print(clip, len(tracks), "tracks, scroll", round(v, 2), "px/frame =", round(-v * FPS * BAR), "px/bar")
def crossings(Xp):
    out = []
    for t in tracks:
        k, b = np.polyfit(t[:, 0], t[:, 1], 1)
        if k > 0.6 * v: continue                       # not scrolling with the page
        f = (Xp - b) / k
        if t[0, 0] - 3 <= f <= t[-1, 0] + 3:
            D = Dp(f / FPS)
            if D is not None: out.append((round(f / FPS + D + OF, 3), int(t[0, 2]), round(f / FPS, 3)))
    return sorted(out)
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + fl.std(), distance=8); on = T[pk]
def score(Xp):
    h = crossings(Xp); d = np.array([abs(x[0] - on[np.argmin(abs(on - x[0]))]) for x in h])
    return (d < 0.03).sum() - 0.5 * (d >= 0.06).sum(), h
if XP is None:
    XP = max(np.arange(572, 1885, 2), key=lambda x: score(x)[0])
h = crossings(XP); d = [(x[0] - on[np.argmin(abs(on - x[0]))]) * 1000 for x in h]
print(f"Xp {XP:.0f}: {len(h)} hits, median offset {np.median(d):.1f} ms, within 30 ms {sum(abs(x) < 30 for x in d)}")
for x, dd in zip(h, d): print(f"  song {x[0]:7.3f} bar {x[0] / BAR + 1:6.2f} y{x[1]:4d} clip {x[2]:6.3f} {dd:+6.0f} ms")
json.dump(h, open(f"{clip}_hits.json", "w")); json.dump({"Xp": float(XP), "v": float(v)}, open(f"{clip}_follow.json", "w"))
