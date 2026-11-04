"""Note hits from the picture (MOVE, purple theme, 60 fps OBS). For every note track (same lane, continuous left edge),
fit rel = note_x0 - playhead_x over the frames approaching the playhead and extrapolate to rel = 0 (works for page mode,
where the playhead moves, and follow mode, where the notes scroll). song = clip + Dp (clock) + OF.
Then compare to master onsets in the instrument's band and report the median offset.
usage: move_validate.py clip band OF  (band: sub|mid|hi)"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band, OF = sys.argv[1], sys.argv[2], float(sys.argv[3])
FPS = 60
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
# playhead: fill short gaps
ph = [s[0] for s in S]
for i in range(len(ph)):
    if ph[i] is None:
        a = next((j for j in range(i - 1, max(-1, i - 6), -1) if S[j][0] is not None), None)
        b = next((j for j in range(i + 1, min(len(S), i + 6)) if S[j][0] is not None), None)
        if a is not None and b is not None and abs(S[b][0] - S[a][0]) < 60:
            ph[i] = S[a][0] + (S[b][0] - S[a][0]) * (i - a) / (b - a)
def Dp(t):
    for a, b, D, *_ in C:
        if a - 0.1 <= t <= b + 0.1: return D
    return None
# tracks: velocity-aware (follow mode scrolls the notes left, page mode keeps them still)
tracks = []          # each: list of (frame, x0, x1, y)
open_ = []
def vel(tr):
    if len(tr) < 2: return 0.0
    f1, a1 = tr[-1][0], tr[-1][1]
    old = [p for p in tr if f1 - p[0] >= 8]
    if not old: return 0.0
    f0, a0 = old[-1][0], old[-1][1]
    return (a1 - a0) / (f1 - f0)
for i, s in enumerate(S):
    nxt = []
    used = set(k for k, (a, b, yy) in enumerate(s[1]) if ph[i] is not None and 0 < a - ph[i] <= 7)   # cut edge right of the playhead
    for tr in sorted(open_, key=lambda t: -len(t)):
        f, x0, x1, y = tr[-1]
        if i - f > 8: tracks.append(tr); continue
        pred = x0 + vel(tr) * (i - f)
        cand = [k for k, (a, b, yy) in enumerate(s[1]) if k not in used and abs(yy - y) <= 4 and abs(a - pred) <= 22]
        if cand:
            k = min(cand, key=lambda k: abs(s[1][k][0] - pred)); used.add(k)
            a, b, yy = s[1][k]; tr.append((i, a, b, yy))
        nxt.append(tr)
    for k, (a, b, yy) in enumerate(s[1]):
        if k not in used: nxt.append([(i, a, b, yy)])
    open_ = nxt
tracks += open_
hits = []
for tr in tracks:
    pts = np.array([(f, x0 - ph[f]) for f, x0, x1, y in tr if ph[f] is not None and x0 > 102], float)
    if len(pts) < 4: continue
    near = pts[(pts[:, 1] > -120) & (pts[:, 1] < 200)]
    if len(near) < 4 or near[:, 1].min() > 12 or near[:, 1].max() < 0: continue
    if not (near[:, 1].max() > 0 and near[:, 1].min() <= 12): continue
    k1, k0 = np.polyfit(near[:, 0], near[:, 1], 1)
    if k1 > -0.3: continue
    res = near[:, 1] - (k1 * near[:, 0] + k0)
    keep = abs(res) < 15
    if keep.sum() >= 4:
        k1, k0 = np.polyfit(near[keep, 0], near[keep, 1], 1)
    fz = -k0 / k1
    if not (near[0, 0] - 6 <= fz <= near[-1, 0] + 14): continue
    t = fz / FPS; D = Dp(t)
    if D is not None:
        hits.append((round(t + D + OF, 3), tr[0][3], round(t, 3)))
hits.sort()
# master onsets
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "75", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "mid": (300, 3000), "hi": (5000, 11000), "low": (100, 400)}[band]
b = np.log1p(Sx[(fr >= lo) & (fr < hi)].sum(0)); fl = np.maximum(0, np.diff(b, prepend=b[0]))
pk, _ = find_peaks(fl, height=fl.mean() + fl.std(), distance=8); on = T[pk]
d = [(h[0] - on[np.argmin(abs(on - h[0]))]) * 1000 for h in hits]
print(clip, len(hits), "hits; median offset vs", band, "onsets", round(float(np.median(d)), 1) if d else None, "ms")
BAR = 240 / 193
for h, dd in zip(hits, d):
    print(f"  song {h[0]:7.3f} bar {h[0] / BAR + 1:6.2f} y{h[1]:4d} clip {h[2]:6.3f}  {dd:+6.0f} ms")
json.dump(hits, open(f"{clip}_hits.json", "w"))
