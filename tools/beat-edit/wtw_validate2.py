"""Walk This Way (beige FL theme, page-mode rolls): note hits in song time. song = clip + Dp(clock) + OFF.
Playhead x interpolated over short gaps; a note counts if seen (w>=3) in >=3 of the previous 12 frames.
Writes validate.json {view: {clip, runs, cross: [(song_t, y, clip_t, err_ms)]}}"""
import subprocess, json, sys, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; OFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
V = {"HAT": ("hat", (7000, 11000)), "OH": ("oh", (5000, 11000)), "TOM": ("tom", (80, 600)), "TOMS": ("toms", (80, 600)), "RIM": ("rim", (1500, 8000))}
C = json.load(open("clock2.json"))
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "110", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def onsets(lo, hi):
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); h = 22
    e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
    fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 94), distance=int(0.05 * SR / h))
    return pk * h / SR
out = {}
for name, (clip, band) in V.items():
    S = json.load(open(clip + "_scan.json")); on = onsets(*band)
    runs = [(a, b, Dp + OFF) for a, b, Dp, *_ in C[clip]["runs"] if b - a >= 0.4]
    def song(t):
        for a, b, D in runs:
            if a - 0.03 <= t <= b + 0.03: return t + D
        return None
    X = [s[0] for s in S]
    for k in range(1, len(X) - 1):           # fill short gaps
        if X[k] is None:
            j0 = next((j for j in range(k - 1, max(-1, k - 7), -1) if X[j] is not None), None)
            j1 = next((j for j in range(k + 1, min(len(X), k + 7)) if X[j] is not None), None)
            if j0 is not None and j1 is not None and 0 <= X[j1] - X[j0] < 30 * (j1 - j0):
                X[k] = X[j0] + (X[j1] - X[j0]) * (k - j0) / (j1 - j0)
    hits = []
    mv = [X[k] - X[k - 1] for k in range(1, len(X)) if X[k] is not None and X[k - 1] is not None and 0.5 < X[k] - X[k - 1] < 10]
    VPF = float(np.median(mv)) if mv else 2.45            # px per frame (page-mode playhead speed = follow scroll speed)
    parked = [k > 10 and X[k] is not None and all(X[j] is not None and abs(X[j] - X[k]) <= 3 for j in range(k - 10, k)) for k in range(len(X))]
    est = []
    for k in range(12, len(S)):
        x1, x = X[k - 1], X[k]
        if x1 is None or x is None: continue
        if parked[k]:                                     # follow mode: notes scroll left into the parked playhead
            for q in S[k][1]:
                if q[2] >= 3 and q[0] > x + 10 and q[0] < x + 400:
                    tc = (k + (q[0] + 1 - x) / VPF) / FPS; s_ = song(tc)
                    if s_ is not None: est.append((s_, q[1], tc))
            continue
        if not (0 < x - x1 < 40): continue
        seen = {}
        for j in range(k - 12, k):
            for q in S[j][1]:
                if q[2] < 3: continue
                key = (round(q[0] / 4), round(q[1] / 6)); seen.setdefault(key, []).append(q)
        for key, qs in seen.items():
            if len(qs) < 3: continue
            qx = float(np.median([q[0] for q in qs])); qy = int(np.median([q[1] for q in qs]))
            if x1 < qx + 1 <= x:
                tc = (k - 1 + (qx + 1 - x1) / (x - x1)) / FPS; s_ = song(tc)
                if s_ is not None: hits.append((round(s_, 3), qy, round(tc, 3)))
    est.sort(); cl = []
    for s_, yy, tc in est:
        for c in cl[-8:]:
            if abs(np.median(c[0]) - s_) < 0.03 and abs(c[1] - yy) <= 6: c[0].append(s_); c[2].append(tc); break
        else: cl.append([[s_], yy, [tc]])
    hits += [(round(float(np.median(c[0])), 3), c[1], round(float(np.median(c[2])), 3)) for c in cl if len(c[0]) >= 6]
    hits.sort(); ded = []
    for h_ in hits:
        if not ded or abs(h_[0] - ded[-1][0]) > 0.03: ded.append(h_)
    cross = [(s, yy, tc, int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))) for s, yy, tc in ded]
    e = np.array([c[3] for c in cross]) if cross else np.zeros(1)
    print(f"{name:5s} hits {len(cross):3d} median err {np.median(e):+.0f} ms |err|<35: {(np.abs(e) < 35).sum()}  song {cross[0][0] if cross else '-'}-{cross[-1][0] if cross else '-'}")
    out[name] = {"clip": clip, "runs": runs, "cross": cross, "miss": []}
json.dump(out, open("validate.json", "w"))
