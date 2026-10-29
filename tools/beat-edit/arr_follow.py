"""Follow-mode rolls (hat, hatpat): playhead parked at x=PX, notes scroll left at V px/s. V from frame-to-frame shifts.
Each note in each frame predicts when its left edge reaches PX; cluster per note. Adds HAT / HATPAT to validate.json."""
import subprocess, json, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; B0 = 0.082
C = json.load(open("clock2.json")); out = json.load(open("validate.json"))
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
e = np.abs(sosfilt(butter(4, [7000, 11000], btype="band", fs=SR, output="sos"), y)); h = 22
e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 90), distance=int(0.05 * SR / h)); on = pk * h / SR
for name, clip in (("HAT", "hat"), ("HATPAT", "hatpat")):
    S = json.load(open(clip + "_scan.json"))
    runs = [(a, b, Dp + B0) for a, b, Dp, *_ in C[clip]["runs"] if b - a >= 0.4]
    def song(t):
        for a, b, D in runs:
            if a <= t <= b: return t + D
        return None
    PX = int(np.median([s[0] for s in S if s[0] is not None]))
    LIM = {'hat': 30, 'hatpat': 85}[clip]
    red = [k for k in range(1, len(S)) if S[k][1] != S[k - 1][1] and song(k / FPS) is not None]
    Vs = []
    for k0, k1 in zip(red, red[1:]):
        sh = [q[0] - r[0] for q in S[k0][1] for r in S[k1][1] if abs(q[1] - r[1]) <= 2 and abs(q[2] - r[2]) <= 1 and 0 < q[0] - r[0] < LIM and q[0] > PX + 10]
        if sh: Vs.append(np.median(sh) / ((k1 - k0) / FPS))
    V = {'hat': 1092 / 2.4, 'hatpat': 180 / 0.15}[clip]   # from the grid: hat 273 px/beat, hatpat 180 px per 16th
    est = []
    for k in red:
        for q in S[k][1]:
            if q[0] > PX + 8: est.append((song(k / FPS) + (q[0] - PX) / V, q[1]))
    est.sort(); cl = []
    for s, yy in est:
        for c in cl:
            if abs(np.median(c[0]) - s) < 0.04 and abs(c[1] - yy) <= 2: c[0].append(s); break
        else: cl.append([[s], yy])
    hits = sorted((round(float(np.median(c[0])), 3), c[1]) for c in cl if len(c[0]) >= 4)
    hits = [(s, yy) for s, yy in hits if any(a + D <= s <= b + D for a, b, D in runs)]
    cross = [(s, yy, 0.0, int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))) for s, yy in hits]
    er = np.array([c[3] for c in cross])
    print(name, "PX", PX, "V px/s", round(V, 1), "hits", len(cross), "median err", np.median(er) if len(er) else None, "|err|<30:", (np.abs(er) < 30).sum())
    print("   ", [(c[0], c[3]) for c in cross][:50])
    out[name] = {"clip": clip, "runs": runs, "cross": cross, "miss": [], "follow": PX}
json.dump(out, open("validate.json", "w"))
