"""Dope Boy: song time of every visible note hit per piano-roll recording (picture mapping from the FL clock:
song = clip + Dp + B0). Follow-mode rolls (snare, 808): play position = keyboard edge x=355, notes scroll at
px_per_bar / BAR px/s; each fully visible note in each frame predicts its hit time; cluster per note.
Page-mode rolls (clap, vocal, snare2): moving playhead crossings (note boxes from 6 frames earlier).
err = nearest master onset in the instrument band - hit (info). Writes validate.json."""
import subprocess, json, numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050; FPS = 60; BAR = 240 / 190; B0 = 0.077; EDGE = 355; PXBAR = 181.3
C = json.load(open("clock2.json"))
VIEWS = {"SNARE": ("snare", "follow", (1500, 6000)), "808": ("s808", "follow", (30, 160)),
         "CLAP": ("clap", "page", (1500, 6000)), "VOX": ("vocal", "page", (300, 4000)), "SNARE2": ("snare2", "page", (1500, 6000))}
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def onsets(lo, hi):
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); h = 22
    e = np.convolve(e, np.ones(h) / h, mode="same")[::h]; le = np.log1p(e * 300)
    fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 96), distance=int(0.06 * SR / h))
    return pk * h / SR
out = {}
for name, (clip, kind, band) in VIEWS.items():
    S = json.load(open(clip + "_scan.json")); on = onsets(*band)
    runs = [(a, b, Dp + B0) for a, b, Dp, *_ in C[clip]["runs"] if b - a >= 0.4]
    def song(t):
        for a, b, D in runs:
            if a <= t <= b: return t + D
        return None
    hits = []
    if kind == "follow":
        V = PXBAR / BAR; est = []
        for k in range(len(S)):
            t = k / FPS
            if song(t) is None: continue
            for q in S[k][1]:
                if q[0] > 372 and q[0] < 1880: est.append((song(t) + (q[0] - EDGE) / V, q[1]))
        est.sort(); cl = []
        for s, yy in est:
            for c in cl:
                if abs(np.median(c[0]) - s) < 0.03 and c[1] == yy: c[0].append(s); break
            else: cl.append([[s], yy])
        for c in cl:
            if len(c[0]) < 20: continue
            s = float(np.median(c[0]))
            # only hits that happen while the recording is still playing (note reaches the edge on screen)
            if any(a + D <= s <= b + D for a, b, D in runs): hits.append((round(s, 3), c[1], 0.0))
    else:
        for k in range(7, len(S)):
            xs = [S[j][0] for j in range(k - 6, k + 1)]
            if any(v is None for v in xs) or any(not (0 <= xs[i + 1] - xs[i] < 60) for i in range(6)): continue
            x1, x = xs[-2], xs[-1]
            if x <= x1: continue
            for q in S[k - 6][1]:
                if x1 < q[0] + 1 <= x:
                    tc = (k - 1 + (q[0] + 1 - x1) / (x - x1)) / FPS; s = song(tc)
                    if s is not None: hits.append((round(s, 3), q[1], round(tc, 3)))
    hits.sort()
    ded = []
    for h_ in hits:
        if not ded or abs(h_[0] - ded[-1][0]) > 0.03 or h_[1] != ded[-1][1]: ded.append(h_)
    cross = [(s, yy, tc, int(round(1000 * float(on[np.argmin(np.abs(on - s))] - s)))) for s, yy, tc in ded]
    e = np.array([c[3] for c in cross]) if cross else np.zeros(0)
    print(f"{name:6s} hits {len(cross):3d} median err {np.median(e) if len(e) else 0:+.0f} ms |err|<35: {(np.abs(e) < 35).sum()}")
    print("   ", [(c[0], c[3]) for c in cross][:70])
    out[name] = {"clip": clip, "runs": runs, "cross": cross}
json.dump(out, open("validate.json", "w"))
