"""Smooth-follow roll with a hidden playhead (CREEPIN hat / second 808; from the MERSEY snare method). The view scrolls
left in jerky steps; track the content position by phase correlation, give every note a content coordinate u, and solve
the playhead x so that note crossings land on master onsets in the band. Also writes {clip}_hits.json.
usage: cr_xp.py clip band [x_lo x_hi]"""
import json, sys, subprocess, numpy as np, cv2
from scipy.signal import stft, find_peaks
clip, band = sys.argv[1], sys.argv[2]
XLO, XHI = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (480, 1880)
BAR = 240 / 193; FPS = 60; OF = 0.02
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
cap = cv2.VideoCapture(clip + ".mp4"); prev = None; pos = [0.0]
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f[230:900, 480:1860], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if prev is not None:
        d = cv2.phaseCorrelate(prev, g)[0][0]; pos.append(pos[-1] + (d if abs(d) < 400 else 0))
    prev = g
pos = np.array(pos[:len(S)]); t = np.arange(len(pos)) / FPS
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 1.5 * fl.std(), distance=8); on = T[pk]
runs = []
for a, b, D, *_ in C:
    m = (t >= a) & (t <= b)
    if m.sum() < 30: continue
    k, c = np.polyfit(t[m] + D + OF, pos[m], 1)
    us = []
    for i in np.where(m)[0]:
        for x0, x1, yy in S[i][1]:
            if yy > 235 and 485 < x0 < 1850: us.append(x0 - pos[i])
    us = np.sort(us); cl = []
    for u in us:
        if cl and u - cl[-1][-1] < 6: cl[-1].append(u)
        else: cl.append([u])
    U = [float(np.median(q)) for q in cl if len(q) >= 8]
    runs.append((a, b, D, k, c, U)); print("run", a, b, "scroll", round(k, 1), "px/s =", round(-k * BAR), "px/bar; notes", len(U))
def crossings(X):
    h = []
    for a, b, D, k, c, U in runs:
        for u in U:
            s = (X - u - c) / k
            if a + D <= s <= b + D: h.append(s)
    return np.array(sorted(h))
res = []
for X in range(XLO, XHI, 2):
    h = crossings(X)
    if len(h) < 3: continue
    d = np.abs(h[:, None] - on[None, :]).min(1); res.append(((d < 0.025).sum() - (d > 0.05).sum(), len(h), X))
res.sort(reverse=True); print("best", res[:6])
X = res[0][2]; h = crossings(X); d = [(x - on[np.argmin(abs(on - x))]) * 1000 for x in h]
print(f"Xp {X}: {len(h)} hits, median {np.median(d):.1f} ms, within 30 ms {sum(abs(x) < 30 for x in d)}")
print([(round(x / BAR + 1, 2), round(dd)) for x, dd in zip(h, d)])
json.dump([(round(float(x), 3), 0, 0) for x in h], open(f"{clip}_hits.json", "w"))
json.dump({"Xp": X, "speed": float(np.median([r[3] for r in runs]))}, open(f"{clip}_follow.json", "w"))
