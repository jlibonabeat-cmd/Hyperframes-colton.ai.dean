"""NO COMPETITION: validate_x.json from the picture hits (<clip>_hits.json: song, y, clip_t, from nc_validate.py with
OF 0.03) plus phrase shifts, each hit kept only where the master has it at the shifted time (band onset flux >= 0.6x
the original's). Also hits.json (kick / snare / hat onsets from the edit start) for the renderer's scoring.
Arrangement check: kick (pattern 4) rests in bars 33-40, hats (pattern 2) stop at bar 38.5 -> shift windows below."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BAR = 240 / 179; START = 0.035 + 24 * BAR; END = START + 24 * BAR
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "90", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); S = np.abs(Z); T = np.arange(S.shape[1]) * 128 / sr
F = {}
for nm, lo, hi in [("sub", 30, 100), ("low", 100, 400), ("snare", 1500, 5000), ("hi", 5000, 11000)]:
    b = np.log1p(S[(fr >= lo) & (fr < hi)].sum(0)); F[nm] = np.maximum(0, np.diff(b, prepend=b[0]))
def fl(k, t):
    i = (T >= t - 0.035) & (T <= t + 0.035)
    return F[k][i].max() if i.any() else 0.0
# view: (hits file, band, shift in bars, song window where the shifted pattern is really playing)
SPEC = {
    "RESO": ("reso", "low", 0, (START, START + 8 * BAR)),
    "KICK": ("kick", "sub", 0, (START, START + 8 * BAR)),
    "KICK_R4": ("kick", "sub", 4, (START, START + 8 * BAR)),
    "KICK_16": ("kick", "sub", 16, (START + 16 * BAR, END)),
    "KICK_20": ("kick", "sub", 20, (START + 16 * BAR, END)),
    "HATS": ("hats", "hi", 0, (START, START + 8 * BAR)),
    "HATS_R4": ("hats", "hi", 4, (START, START + 8 * BAR)),
    "HATS_8": ("hats", "hi", 8, (START + 8 * BAR, START + 13.5 * BAR)),
    "B808": ("b808", "sub", 0, (START + 8 * BAR, START + 16 * BAR)),
    "B808F": ("b808f", "sub", 0, (START + 8 * BAR, START + 16 * BAR)),
    "B808_8": ("b808", "sub", 8, (START + 16 * BAR, END)),
    "B808F_8": ("b808f", "sub", 8, (START + 16 * BAR, END)),
}
out = {}
for v, (f, band, sh, (a, b)) in SPEC.items():
    h = [x for x in json.load(open(f"{f}_hits.json")) if x[0] >= 32.0]      # drop the hat hit from a misread clock run
    keep, drop = [], []
    for t, yy, c in h:
        t2 = t + sh * BAR
        if not a - 0.05 <= t2 < b: continue
        (keep if sh == 0 or fl(band, t2) >= max(0.02, 0.6 * fl(band, t)) else drop).append((t2, yy))
    out[v] = {"cross": keep, "dropped": drop}
    print(f"{v:8s} kept beats", [round((k[0] - START) / (BAR / 4), 2) for k in keep], "dropped", [round((d[0] - START) / (BAR / 4), 2) for d in drop])
json.dump(out, open("validate_x.json", "w"))
H = {}
for nm, band in (("kick", "sub"), ("snare", "snare"), ("hat", "hi")):
    pk, pr = find_peaks(F[band], height=F[band].mean() + F[band].std(), distance=10)
    t = T[pk] - START; m = (t >= 0) & (t < 24 * BAR + 1)
    H[nm] = {"t": t[m].round(4).tolist(), "s": (pr["peak_heights"][m] / pr["peak_heights"].max()).round(3).tolist()}
json.dump(H, open("hits.json", "w"))
