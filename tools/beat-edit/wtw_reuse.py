"""Phrase-repeat reuse for Walk This Way: shift each validated note hit by 4 bars (_R, bars 25-28) and 8 bars (_8, the
drop, bars 29-32, which plays an exported audio bounce of the same drum patterns) and keep only the hits the master
actually has at the shifted time (onset flux in the instrument's band within +-35 ms at least 0.6x the original's).
Writes validate_x.json = validate.json + *_R + *_8 entries."""
import json, subprocess, sys, numpy as np
from scipy.signal import stft
HERE = sys.argv[1]
BAR = 4 * 60 / 98
v = json.load(open(f"{HERE}/validate.json"))
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", "40", "-t", "45", "-i", f"{HERE}/beat.mp3", "-ac", "1",
                                  "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); S = np.abs(Z); T = 40 + np.arange(S.shape[1]) * 128 / sr
F = {}
for nm, lo, hi in [("hi", 6000, 11000), ("tom", 90, 350)]:
    b = np.log1p(S[(fr >= lo) & (fr < hi)].sum(0)); F[nm] = np.maximum(0, np.diff(b, prepend=b[0]))
BAND = {"HAT": "hi", "OH": "hi", "RIM": "hi", "TOM": "tom"}
def fl(k, t):
    i = (T >= t - 0.035) & (T <= t + 0.035)
    return F[BAND[k]][i].max() if i.any() else 0.0
def heard(k, t0, t):
    return fl(k, t) >= max(0.03, 0.6 * fl(k, t0))
out = dict(v)
for k in BAND:
    for suf, nb, lim in (("_R", 4, 49.0 + 32 * BAR / 4), ("_8", 8, 49.0 + 48 * BAR / 4)):
        keep, drop = [], []
        for c in v[k]["cross"]:
            t = c[0] + nb * BAR
            if t >= lim: continue
            (keep if heard(k, c[0], t) else drop).append((t,) + tuple(c[1:]))
        out[k + suf] = {"cross": keep, "dropped": drop}
        print(k + suf, len(keep), "kept", len(drop), "dropped", [round((d[0] - 49.0) / (BAR / 4), 2) for d in drop])
json.dump(out, open(f"{HERE}/validate_x.json", "w"))
