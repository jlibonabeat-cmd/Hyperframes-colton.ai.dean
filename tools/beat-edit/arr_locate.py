"""For each playback run (clock2.json) of each recording: where in the master does its own audio line up?
song = clip + L. Band-limited spectral flux, 5 ms hop, lags over master 0-60 s. Top peaks per run.
Writes locate.json {clip: [{a,b,Dp,peaks:[(corr,L)]}]}"""
import subprocess, json, sys, numpy as np
SR = 22050; HOP = 110
def load(p):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def flux(y, lo, hi):
    w = 1024; n = (len(y) - w) // HOP
    idx = np.arange(w)[None, :] + HOP * np.arange(n)[:, None]
    X = np.abs(np.fft.rfft(y[idx] * np.hanning(w), axis=1)); F = np.fft.rfftfreq(w, 1 / SR); m = (F >= lo) & (F < hi)
    L = np.log1p(X[:, m] * 20); f = np.concatenate([[0], np.maximum(0, np.diff(L, axis=0)).sum(1)])
    return f - np.convolve(f, np.ones(60) / 60, mode="same")
master = load("beat.mp3")[:SR * 62]
C = json.load(open("clock2.json")); res = {}
BAND = (30, 11000)
fm = flux(master, *BAND)
for clip in sys.argv[1:]:
    fy = flux(load(clip + ".mp4"), *BAND); out = []
    for a, b, Dp, *_ in C[clip]["runs"]:
        if b - a < 1.5: continue
        seg = fy[int(a * 200):int(b * 200)]; seg = (seg - seg.mean()) / (seg.std() + 1e-9)
        n = len(seg); cs = []
        for j in range(0, len(fm) - n):
            r = fm[j:j + n]; cs.append((float((seg * (r - r.mean())).mean() / (r.std() + 1e-9)), j))
        cs.sort(reverse=True); peaks = []
        for c, j in cs:
            if all(abs(j - p[1]) > 60 for p in peaks): peaks.append((c, j))
            if len(peaks) == 4: break
        out.append({"a": a, "b": b, "Dp": Dp, "peaks": [(round(c, 3), round(j / 200 - a, 3)) for c, j in peaks]})
        print(clip, a, b, "Dp", round(Dp, 3), [(round(c, 2), round(j / 200 - a, 3)) for c, j in peaks], flush=True)
    res[clip] = out
json.dump(res, open("locate.json", "w"))
