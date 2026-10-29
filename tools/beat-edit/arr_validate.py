"""ARRESTED: every playhead/note crossing per piano-roll recording, in song time (song = clip + Dp + B0 from the
frame-exact FL clock). Note boxes from 6 frames earlier (so the playhead's own pixels never count).
'miss' = notes of the final pattern (last 90 frames) crossed while not drawn yet (putting-in clips).
err = nearest master onset in the instrument band minus crossing (info). Writes validate.json."""
import subprocess, json, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; B0 = 0.082
V = {"CLAP": ("clap", (1500, 6000)), "HAT": ("hat", (7000, 11000)), "808": ("s808", (30, 160)), "OH1": ("oh1", (6000, 11000)),
     "OH2": ("oh2", (6000, 11000)), "RIM": ("rim", (1500, 8000)), "HATPAT": ("hatpat", (7000, 11000))}
C = json.load(open("clock2.json"))
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def onsets(lo, hi):
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); h = 22
    e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
    fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 96), distance=int(0.06 * SR / h))
    return pk * h / SR
out = {}
for name, (clip, band) in V.items():
    S = json.load(open(clip + "_scan.json")); on = onsets(*band)
    runs = [(a, b, Dp + B0) for a, b, Dp, *_ in C[clip]["runs"] if b - a >= 0.4]
    def song(t):
        for a, b, D in runs:
            if a - 0.05 <= t <= b + 0.05: return t + D
        return None
    fin = []
    for x, bs in S[-90:]:
        for q in bs:
            if not any(abs(q[0] - r[0]) <= 6 and abs(q[1] - r[1]) <= 6 for r in fin): fin.append(q)
    cross, miss = [], []
    for k in range(7, len(S)):
        xs = [S[j][0] for j in range(k - 6, k + 1)]
        if any(v is None for v in xs) or any(not (0 <= xs[i + 1] - xs[i] < 60) for i in range(6)): continue
        x1, x = xs[-2], xs[-1]
        if x <= x1: continue
        shown = S[k - 6][1]
        for q in shown:
            if x1 < q[0] + 1 <= x:
                tc = (k - 1 + (q[0] + 1 - x1) / (x - x1)) / FPS; s = song(tc)
                if s is not None: cross.append((round(s, 3), q[1], round(tc, 3)))
        for q in fin:
            if x1 < q[0] + 1 <= x and not any(abs(q[0] - r[0]) <= 6 and abs(q[1] - r[1]) <= 6 for r in shown):
                s = song((k - 0.5) / FPS)
                if s is not None: miss.append((round(s, 3), q[1]))
    ded = []
    for c in sorted(cross):
        if not ded or abs(c[0] - ded[-1][0]) > 0.03 or c[1] != ded[-1][1]: ded.append(c)
    cross = [(s, yy, tc, int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))) for s, yy, tc in ded]
    e = np.array([c[3] for c in cross]) if cross else np.zeros(0)
    print(f"{name:6s} hits {len(cross):3d} median err {np.median(e) if len(e) else 0:+.0f} ms |err|<35: {(np.abs(e) < 35).sum()}  miss {len(miss)}")
    print("   ", [(c[0], c[3]) for c in cross][:60]); print("    miss", [m[0] for m in miss][:30])
    out[name] = {"clip": clip, "runs": runs, "cross": cross, "miss": miss}
json.dump(out, open("validate.json", "w"))
