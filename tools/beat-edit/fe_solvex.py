"""Solve the play line of a smooth-follow roll from the picture itself. pos(t) = measured scroll (phase correlation,
fe_pos), each note gets a content coordinate u = x0 - pos; it plays when pos(t) = X - u. Score every X by how many of
the note crossings land within 15 ms of a master onset in the band. usage: fe_solvex.py clip band [x_lo x_hi]"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band = sys.argv[1], sys.argv[2]
XLO, XHI = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (575, 1880)
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
pos = np.array(json.load(open(f"{clip}_pos.json"))); t = np.arange(len(pos)) / 60
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 0.8 * fl.std(), distance=8); on = T[pk]
best_all = []
for a, b, D, *_ in C:
    m = np.where((t >= a) & (t <= b))[0]
    if len(m) < 60: continue
    us = []
    for i in m:
        for x0, x1, yy in S[i][1]:
            if yy > 195 and 580 < x0 < 1875: us.append(x0 - pos[i])
    us = np.sort(us); cl = []
    for u in us:
        if cl and u - cl[-1][-1] < 4: cl[-1].append(u)
        else: cl.append([u])
    U = np.array([np.median(q) for q in cl if len(q) >= 6])
    P = pos[m]; Ts = t[m] + D + 0.02
    order = np.argsort(P); Ps, Tt = P[order], Ts[order]     # pos decreases with time; invert by interpolation
    res = []
    for X in range(XLO, XHI):
        target = X - U
        ok = (target >= Ps.min()) & (target <= Ps.max())
        if ok.sum() < 3: continue
        s = np.interp(target[ok], Ps, Tt)
        d = np.abs(s[:, None] - on[None, :]).min(1)
        res.append(((d < 0.015).mean(), ok.sum(), X, float(np.median(np.array([(x - on[np.argmin(abs(on - x))]) for x in s])) * 1000)))
    res.sort(reverse=True)
    print(f"{clip} run {a}-{b} ({len(U)} notes): best", [(round(r[0], 2), r[1], r[2], round(r[3])) for r in res[:6]])
