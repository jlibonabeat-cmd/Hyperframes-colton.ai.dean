"""Sync check: picture note-ons -> song time (clock D + 0.02 picture lead) vs master onsets in the element's band.
Page rolls: a note's left edge crossing FL's playhead. Hidden-playhead follow rolls: a note's left edge reaching the
grid edge (x 92) / its key lighting up. usage: agt_sync.py clip lo hi [edge]"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); edge = len(sys.argv) > 4
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
b = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0) * 10); fl = np.maximum(0, np.diff(b, prepend=b[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 0.7 * fl.std(), distance=6); on = T[pk]
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
def D(t): return next((r[2] for r in C if r[0] <= t <= r[1]), None)
ev = []
for i in range(1, len(S)):
    if edge:
        a = {(round(bb[2] / 4)): bb[0] for bb in S[i - 1][1] if bb[0] <= 95}
        c = {(round(bb[2] / 4)): bb[0] for bb in S[i][1] if bb[0] <= 95}
        # a note's left edge reaches the grid edge when a row first gets a box starting at the edge (x<=95)
        for k in c:
            if k not in a:
                prev = [bb for bb in S[i - 1][1] if round(bb[2] / 4) == k and 95 < bb[0] < 160]
                if prev:
                    x0 = prev[0][0]; x1 = 92; sp = None
                    t = (i - 1 + (x0 - 92) / max(1, x0 - 92 + 1)) / 60
                    ev.append((i / 60, k * 4))
    else:
        p0, p1 = S[i - 1][0], S[i][0]
        if p0 is None or p1 is None or abs(p1 - p0) > 40: continue
        for x0_, _, y0 in S[i - 1][1]:
            for x1_, _, y1 in S[i][1]:
                if abs(y0 - y1) <= 3 and abs(x0_ - x1_) < 14:
                    d0, d1 = x0_ - p0, x1_ - p1
                    if d0 > 0 >= d1: ev.append(((i - 1 + d0 / (d0 - d1)) / 60, y1))
out = []
for t, yy in sorted(set(ev)):
    d = D(t)
    if d is None: continue
    s = t + d + 0.02; out.append((s, yy))
dd = np.array([(s - on[np.argmin(abs(on - s))]) * 1000 for s, _ in out]) if out else np.array([np.nan])
BT = 60 / 140
print(f"{clip}: {len(out)} note-ons; vs master {lo:.0f}-{hi:.0f} Hz onsets: median {np.nanmedian(dd):+.0f} ms, |d|<30 ms {(abs(dd) < 30).mean():.2f}")
print("   ", [(round(s / BT, 2), yy, int(d)) for (s, yy), d in zip(out, dd)][:40])
json.dump(out, open(f"{clip}_cross.json", "w"))
