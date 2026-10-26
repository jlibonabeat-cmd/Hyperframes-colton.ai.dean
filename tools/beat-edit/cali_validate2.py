"""CALI MAN: every playhead/note crossing per recording in song (mp3) time.
Picture mapping from the frame-exact FL clock (clock2.json): song = clip + Dp + B0 (+ 32 bars for pattern-mode clips).
A crossing = the playhead passes a note's left edge, using note boxes from 6 frames earlier on the same page (so the
playhead's own split fragments never count). err = nearest master onset in the instrument band minus crossing (info).
Writes validate.json {view: {clip, runs, cross: [(song_t, y, clip_t, err_ms)]}}."""
import subprocess, json, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; BAR = 240 / 190; B0 = 0.015; P = B0 + 32 * BAR
# continuous playback runs (clip_a, clip_b, D): checked against the raw clock (OCR misreads '4:xx' as '40x', not gaps)
RUNS = {"808A": ("s808a", [(2.3, 11.7, -2.288 + P)], (30, 160)),
        "808B": ("s808b", [(7.9, 17.35, -7.884 + P)], (30, 160)),
        "CLAP": ("clap", [(7.15, 15.45, -7.132 + P)], (1200, 6000)),
        "HAT": ("hat", [(1.5, 6.8, -0.992 + P), (7.6, 10.45, -0.993 + P)], (7000, 11000)),
        "KICK": ("kick", [(3.45, 7.05, 37.021 + B0), (7.75, 12.25, 37.01 + B0)], (30, 160))}
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def onsets(lo, hi):
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); h = 22
    e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
    fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 97), distance=int(0.06 * SR / h))
    return pk * h / SR
out = {}
for name, (clip, runs, band) in RUNS.items():
    S = json.load(open(clip + "_scan.json")); on = onsets(*band)
    def song(t):
        for a, b, D in runs:
            if a <= t <= b: return t + D
        return None
    cross = []
    def shift(b0, b1):
        sh = [q[0] - r[0] for q in b0 for r in b1 if abs(q[1] - r[1]) <= 2 and abs(q[2] - r[2]) <= 2 and 0 <= q[0] - r[0] <= 80]
        return float(np.median(sh)) if len(sh) >= 2 else None
    for k in range(7, len(S)):
        xs = [S[j][0] for j in range(k - 6, k + 1)]
        if any(v is None for v in xs): continue
        x1, x = xs[-2], xs[-1]
        if all(abs(xs[i + 1] - xs[i]) <= 2 for i in range(6)):          # follow view: playhead parked, notes scroll left
            if S[k][1] == S[k - 1][1]: continue                           # FL redraws ~every 2-3 OBS frames
            kp = k - 1
            while kp > 0 and S[kp - 1][1] == S[kp][1]: kp -= 1           # first frame of the previous redraw
            kp -= 1
            if kp < 0 or S[kp][0] is None: continue
            sh = [r[0] - q[0] for q in S[k][1] for r in S[kp][1] if abs(q[1] - r[1]) <= 2 and abs(q[2] - r[2]) <= 2
                  and q[0] > x + 8 and 0 < r[0] - q[0] <= 120]
            if len(sh) < 1: continue
            v = float(np.median(sh)); dk = k - kp
            for q in S[k][1]:
                if abs(q[0] + q[2] - (x - 2)) <= 2 and x - v - 1 <= q[0] <= x:     # left part of a note that just reached the playhead
                    tc = (k - (x - q[0]) / v * dk) / FPS; s = song(tc)
                    if s is not None: cross.append((round(s, 3), q[1], round(tc, 3), int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))))
            continue
        if any(not (0 <= xs[i + 1] - xs[i] < 60) for i in range(6)) or x <= x1: continue   # page view: playhead moves
        for q in S[k - 6][1]:
            if x1 < q[0] + 1 <= x:
                tc = (k - 1 + (q[0] + 1 - x1) / (x - x1)) / FPS; s = song(tc)
                if s is not None: cross.append((round(s, 3), q[1], round(tc, 3), int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))))
    cross.sort()
    # de-dup (same time, stacked notes) -> keep one per 25 ms with all ys merged into the y range later
    e = np.array([c[3] for c in cross]) if cross else np.zeros(0)
    print(f"{name:5s} crossings {len(cross):3d}  median err {np.median(e) if len(e) else 0:+.0f} ms  |err|<40: {(np.abs(e) < 40).sum()}")
    print("   ", [(c[0], c[3]) for c in cross][:60])
    out[name] = {"clip": clip, "runs": runs, "cross": cross}
json.dump(out, open("validate.json", "w"))
