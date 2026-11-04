"""RACK ON TOP: validate_x.json = validate.json + phrase-shifted views, each shifted hit kept only where the master has
it (band onset flux within +-35 ms >= 0.6x the original's). Shifts found by testing every view at +4..+32 bars:
BONGO +12/+28, OH1 +8/+20, 808 +4, 808B +4/+8 (drop 2 changes the first bar of each 4-bar 808 phrase: beats 108.5 110.5
124.5 126.5 from 0:10.02 are NOT heard -> plans keep those crossings off screen)."""
import json, subprocess, numpy as np
from scipy.signal import stft
BAR = 240 / 144; BEAT = BAR / 4; START = 0.02 + 6 * BAR
v = json.load(open("validate.json"))
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "80", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); S = np.abs(Z); T = np.arange(S.shape[1]) * 128 / sr
F = {}
for nm, lo, hi in [("sub", 30, 100), ("mid", 300, 3000), ("hi", 5000, 11000)]:
    b = np.log1p(S[(fr >= lo) & (fr < hi)].sum(0)); F[nm] = np.maximum(0, np.diff(b, prepend=b[0]))
def fl(k, t):
    i = (T >= t - 0.035) & (T <= t + 0.035)
    return F[k][i].max() if i.any() else 0.0
SPEC = [("OH1", "hi", 4), ("OH2", "hi", 4), ("BONGO", "mid", 12), ("BONGO", "mid", 28), ("OH1", "hi", 8), ("OH1", "hi", 20), ("808", "sub", 4), ("808", "sub", 8),
        ("808B", "sub", 4), ("808B", "sub", 8)]
out = dict(v)
for k, band, sh in SPEC:
    name = f"{k}_{sh}" if not (k in ("808", "808B") and sh == 8) else k + "_R"
    keep, drop = [], []
    for c in v[k]["cross"]:
        t = c[0] + sh * BAR
        if not 0 <= (t - START) / BEAT < 136: continue
        (keep if fl(band, t) >= max(0.02, 0.6 * fl(band, c[0])) else drop).append((t,) + tuple(c[1:]))
    out[name] = {"cross": keep, "dropped": drop}
    print(f"{name:8s} kept {sorted(set(round((k_[0] - START) / BEAT, 1) for k_ in keep))} dropped {sorted(set(round((d[0] - START) / BEAT, 1) for d in drop))}")
out["OH1_W"] = v["OH1"]
json.dump(out, open("validate_x.json", "w"))
