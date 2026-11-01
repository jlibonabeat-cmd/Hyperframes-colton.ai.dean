"""Packman follow / smooth-scroll rolls: song = clip + Dp(clock) + OFF (+ 24 bars for pattern-mode recordings of
pattern 2). Notes scroll at V = px_per_bar / BAR (ruler) and play at screen x = X. Scan X to check where the notes
are actually heard, then write the hits for the chosen X into validate.json."""
import json, sys, subprocess, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; BAR = 1.2; OFF = 0.015; PAT2 = 24 * BAR
VIEWS = {"HAT4": ("hat4", 1184, (7000, 11000), 0), "HAT2": ("hat2", 414, (7000, 11000), 0), "BELLF": ("bellfull", 169, (600, 2500), 0),
         "LAZER": ("lazer", 196, (1500, 8000), PAT2), "808F2": ("s808_2full", 196, (30, 160), PAT2)}
FIX = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
C = json.load(open("clock2.json")); out = json.load(open("validate.json"))
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
for name, (clip, PXB, (lo, hi), EX) in VIEWS.items():
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); h = 22
    e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
    fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 93), distance=int(0.05 * SR / h)); on = pk * h / SR
    S = json.load(open(clip + "_scan.json")); V = PXB / BAR
    runs = [(a, b, Dp + OFF + EX) for a, b, Dp, *_ in C[clip]["runs"] if b - a >= 0.4]
    def song(t):
        for a, b, D in runs:
            if a - 0.03 <= t <= b + 0.03: return t + D
        return None
    red = [k for k in range(1, len(S)) if S[k][1] != S[k - 1][1] and song(k / FPS) is not None]
    def hits(X):
        est = []
        for k in red:
            for q in S[k][1]:
                if q[0] > X + 6: est.append((song(k / FPS) + (q[0] - X) / V, q[1], k / FPS + (q[0] - X) / V))
        est.sort(); cl = []
        for s, yy, tc in est:
            for c in cl[-8:]:
                if abs(np.median(c[0]) - s) < 0.03 and abs(c[1] - yy) <= 3: c[0].append(s); c[2].append(tc); break
            else: cl.append([[s], yy, [tc]])
        return sorted((round(float(np.median(c[0])), 3), c[1], round(float(np.median(c[2])), 3)) for c in cl if len(c[0]) >= 3 and song(float(np.median(c[2]))) is not None)
    row = []
    for X in range(400, 700, 5):
        hs = hits(X)
        if len(hs) < 3: continue
        d = np.array([on[np.argmin(np.abs(on - s))] - s for s, _, _ in hs]); row.append((round((np.abs(d) < 0.03).mean(), 2), X, len(hs), int(np.median(d) * 1000)))
    print(name, "top X:", sorted(row, reverse=True)[:5])
    if name in FIX:
        hs = hits(FIX[name]); cr = [(s, yy, tc, int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))) for s, yy, tc in hs]
        out[name] = {"clip": clip, "runs": runs, "cross": cr, "miss": [], "follow": FIX[name]}
        e_ = np.array([c[3] for c in cr]) if cr else np.zeros(1); print("   ->", name, "X", FIX[name], "hits", len(cr), "median err", int(np.median(e_)), "|err|<35", int((np.abs(e_) < 35).sum()))
json.dump(out, open("validate.json", "w"))
