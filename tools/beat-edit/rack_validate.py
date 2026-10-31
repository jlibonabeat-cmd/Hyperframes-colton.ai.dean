"""RACK ON TOP: note hits per piano-roll recording in song time. song = clip + Dp + OFF (Dp from the frame-exact FL clock,
OFF fitted once against master onsets). Page views: moving-playhead crossings (note boxes from 6 frames earlier).
Follow views (playhead parked): notes scroll at V = px_per_bar / BAR (from the ruler), each note predicts its hit.
'miss' = final-pattern notes crossed while not drawn yet. Writes validate.json"""
import subprocess, json, sys, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; BAR = 240 / 144; OFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
V = {"BASS": ("bass", "page", (30, 400)), "BONGO": ("bongo", "page", (150, 2000)), "OH1": ("oh1", "page", (6000, 11000)),
     "OH2": ("oh2", "page", (6000, 11000)), "808": ("s808", "page", (30, 160)),
     "HAT": ("hat", 590 / BAR, (7000, 11000)), "808B": ("s808b", 685 / BAR, (30, 160))}
C = json.load(open("clock2.json"))
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "80", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def onsets(lo, hi):
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); h = 22
    e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
    fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 95), distance=int(0.06 * SR / h))
    return pk * h / SR
out = {}
for name, (clip, kind, band) in V.items():
    S = json.load(open(clip + "_scan.json")); on = onsets(*band)
    runs = [(a, b, Dp + OFF) for a, b, Dp, *_ in C[clip]["runs"] if b - a >= 0.4]
    def song(t):
        for a, b, D in runs:
            if a - 0.03 <= t <= b + 0.03: return t + D
        return None
    hits, miss = [], []
    if kind == "page":
        fin = []
        for x, bs in S[-120:]:
            for q in bs:
                if not any(abs(q[0] - r[0]) <= 6 and abs(q[1] - r[1]) <= 6 for r in fin): fin.append(q)
        for k in range(7, len(S)):
            xs = [S[j][0] for j in range(k - 6, k + 1)]
            if any(v is None for v in xs) or any(not (0 <= xs[i + 1] - xs[i] < 60) for i in range(6)): continue
            x1, x = xs[-2], xs[-1]
            if x <= x1: continue
            shown = S[k - 6][1]
            for q in shown:
                if x1 < q[0] + 1 <= x:
                    tc = (k - 1 + (q[0] + 1 - x1) / (x - x1)) / FPS; s = song(tc)
                    if s is not None: hits.append((round(s, 3), q[1], round(tc, 3)))
            for q in fin:
                if x1 < q[0] + 1 <= x and not any(abs(q[0] - r[0]) <= 6 and abs(q[1] - r[1]) <= 6 for r in shown):
                    s = song((k - 0.5) / FPS)
                    if s is not None: miss.append((round(s, 3), q[1]))
    else:
        Vs = kind; PX = int(np.median([s[0] for s in S if s[0] is not None])); est = []
        for k in range(1, len(S)):
            if S[k][1] == S[k - 1][1] or S[k][0] is None: continue
            s0 = song(k / FPS)
            if s0 is None: continue
            for q in S[k][1]:
                if q[0] > PX + 8: est.append((s0 + (q[0] - PX) / Vs, q[1], k / FPS + (q[0] - PX) / Vs))
        est.sort(); cl = []
        for s, yy, tc in est:
            for c in cl:
                if abs(np.median(c[0]) - s) < 0.03 and abs(c[1] - yy) <= 3: c[0].append(s); c[2].append(tc); break
            else: cl.append([[s], yy, [tc]])
        for c in cl:
            if len(c[0]) < 4: continue
            s = float(np.median(c[0])); tc = float(np.median(c[2]))
            if song(tc) is not None: hits.append((round(s, 3), c[1], round(tc, 3)))
    hits.sort(); ded = []
    for h_ in hits:
        if not ded or abs(h_[0] - ded[-1][0]) > 0.03 or h_[1] != ded[-1][1]: ded.append(h_)
    cross = [(s, yy, tc, int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))) for s, yy, tc in ded]
    e = np.array([c[3] for c in cross]) if cross else np.zeros(0)
    print(f"{name:6s} hits {len(cross):3d} median err {np.median(e) if len(e) else 0:+.0f} ms |err|<35: {(np.abs(e) < 35).sum()} miss {len(miss)}  song {cross[0][0] if cross else '-'}-{cross[-1][0] if cross else '-'}")
    out[name] = {"clip": clip, "runs": runs, "cross": cross, "miss": miss, "follow": kind != "page"}
json.dump(out, open("validate.json", "w"))
