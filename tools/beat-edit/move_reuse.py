"""MOVE: build validate_x.json from the picture hits (<clip>_hits.json: song, y, clip_t) for each roll view, plus *_8
(+8 bars = bars 21-28) keeping only hits the master has at the shifted time (band onset flux >= 0.6x the original's).
Also writes hits.json (master kick / snare / hat onsets relative to the edit start) for the renderer's scoring."""
import json, subprocess, sys, numpy as np
from scipy.signal import stft, find_peaks
BAR = 240 / 143; START = 10 * BAR + 0.03
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "80", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); S = np.abs(Z); T = np.arange(S.shape[1]) * 128 / sr
F = {}
for nm, lo, hi in [("sub", 30, 100), ("mid", 300, 3000), ("snare", 1500, 5000), ("hi", 7000, 11000)]:
    b = np.log1p(S[(fr >= lo) & (fr < hi)].sum(0)); F[nm] = np.maximum(0, np.diff(b, prepend=b[0]))
def fl(k, t):
    i = (T >= t - 0.035) & (T <= t + 0.035)
    return F[k][i].max() if i.any() else 0.0
BAND = {"SPINZ": "sub", "PHANTOM": "sub", "SNARE": "mid", "CLAP": "mid", "CHOPS": "mid"}
out = {}
for v, band in BAND.items():
    h = json.load(open(f"{v.lower()}_hits.json"))
    out[v] = {"cross": [(t, yy) for t, yy, c in h]}
    keep, drop = [], []
    for t, yy, c in h:
        (keep if fl(band, t + 8 * BAR) >= max(0.02, 0.6 * fl(band, t)) else drop).append((t + 8 * BAR, yy))
    out[v + "_8"] = {"cross": keep, "dropped": drop}
    print(v, len(h), "hits;", v + "_8", len(keep), "kept", [round((d[0] - START) / (BAR / 4), 2) for d in drop], "dropped (beats)")
# SPINZ_M4: the follow-along part of the Spinz recording (clip >= 7.5) shown 4 bars early (pattern repeats every 4 bars)
h = [x for x in json.load(open("spinz_hits.json")) if x[2] >= 7.5]
keep = [(t - 4 * BAR, yy) for t, yy, c in h if fl("sub", t - 4 * BAR) >= max(0.02, 0.6 * fl("sub", t))]
out["SPINZ_M4"] = {"cross": keep}
print("SPINZ_M4", len(keep), "of", len(h), "kept")
json.dump(out, open("validate_x.json", "w"))
H = {}
for nm, band in (("kick", "sub"), ("snare", "snare"), ("hat", "hi")):
    pk, pr = find_peaks(F[band], height=F[band].mean() + F[band].std(), distance=10)
    t = T[pk] - START; m = (t >= 0) & (t < 18 * BAR + 1)
    H[nm] = {"t": t[m].round(4).tolist(), "s": (pr["peak_heights"][m] / pr["peak_heights"].max()).round(3).tolist()}
json.dump(H, open("hits.json", "w"))
