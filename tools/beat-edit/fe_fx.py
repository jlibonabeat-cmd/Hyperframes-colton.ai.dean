"""FEAR smooth-follow rolls: the scroll speed v and the bar-line phase come from the black bar lines (fe bar-line fit),
so a note's hit time inside the bar is exact; the playhead x is only ambiguous by whole bars (X = p + k*v*BAR).
Pick k by matching note hits against master onsets in the band. Writes {clip}_hits.json and {clip}_follow.json.
usage: fe_fx.py clip band v phase"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band, v, p = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
BAR = 240 / 110; FPS = 60; OF = 0.02; P = v * BAR
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 1.0 * fl.std(), distance=8); on = T[pk]
def hits(X):
    hs = []
    for i, (_, bx) in enumerate(S):
        t = i / FPS; D = next((r[2] for r in C if r[0] <= t <= r[1]), None)
        if D is None: continue
        for x0, x1, yy in bx:
            if yy > 195 and 575 < x0 < 1880 and x0 > X - 2:      # a note left edge still right of the playhead
                hs.append((t + D + OF + (x0 - X) / v, yy))
    hs.sort(); out = []
    for h in hs:                                                # one hit per (note, row): cluster within 40 ms
        if out and abs(h[0] - out[-1][0][-1]) < 0.04 and abs(h[1] - out[-1][1]) < 6: out[-1][0].append(h[0])
        else: out.append(([h[0]], h[1]))
    res = [(float(np.median(a)), int(yy)) for a, yy in out if len(a) >= 6]
    # only hits that actually happen while the recording plays
    return [(t, yy) for t, yy in res if any(r[0] + r[2] <= t <= r[1] + r[2] for r in C)]
best = []
for k in range(-6, 10):
    X = p + k * P
    if not 575 <= X <= 1880: continue
    h = hits(X)
    if not h: continue
    d = np.array([(t - on[np.argmin(abs(on - t))]) * 1000 for t, _ in h])
    best.append(((abs(d) < 25).mean(), len(h), X, float(np.median(d))))
    print(f"X {X:7.1f}: {len(h)} hits, {(abs(d) < 25).sum()} within 25 ms, median {np.median(d):+.1f} ms")
sc, n, X, md = max(best)
h = hits(X); print("chosen X", round(X, 1))
json.dump([(round(t, 3), yy, 0) for t, yy in h], open(f"{clip}_hits.json", "w"))
json.dump({"Xp": X, "v": v}, open(f"{clip}_follow.json", "w"))
