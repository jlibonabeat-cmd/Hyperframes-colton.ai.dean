"""DRUG FIT: (1) per-bar band levels of the master (structure), (2) key-light onsets of each roll vs master onsets in
its band (sync check), (3) note-edit windows (note pixels appearing/disappearing away from the playhead)."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 181.0; BAR = 240 / BPM; sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
BANDS = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}
def band(k): lo, hi = BANDS[k]; return np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0))
def ons(k):
    bb = band(k); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
    pk, _ = find_peaks(fl, height=fl.mean() + 0.8 * fl.std(), distance=8); return T[pk]
print("bar  sub  low  mid  snr  hi  (dB rel. max)")
L = {k: 20 * np.log10(np.maximum(1e-6, [Sx[(fq >= BANDS[k][0]) & (fq < BANDS[k][1])][:, (T >= b * BAR) & (T < (b + 1) * BAR)].mean() for b in range(int(T[-1] / BAR))])) for k in BANDS}
for b in range(len(L["sub"])):
    print(f"{b + 1:3d} " + " ".join(f"{L[k][b] - L[k].max():5.1f}" for k in ("sub", "low", "mid", "snr", "hi")))
C = json.load(open("clock2.json")); out = {}
for clip, bk in [("snare", "snr"), ("snfill", "snr"), ("clap", "snr"), ("ctm808", "sub"), ("ohat", "hi"), ("m808", "sub"), ("spinz", "sub")]:
    S = json.load(open(f"{clip}_scan.json")); on = ons(bk); runs = C[clip]["runs"]; lo = []
    for i in range(1, len(S)):
        if S[i][2] and not S[i - 1][2]:
            t = i / 60; D = next((r[2] for r in runs if r[0] <= t <= r[1]), None)
            if D is not None: lo.append(t + D + 0.02)
    d = np.array([(t - on[np.argmin(abs(on - t))]) * 1000 for t in lo]) if lo else np.array([0.0])
    prev = None; ev = []
    for i, (p, bx, _) in enumerate(S):
        st = {(round(a / 6), round(yy / 6)) for a, b, yy in bx}
        if p is not None: st = {k for k in st if abs(k[0] * 6 - p) > 30}
        if prev is not None:
            ch = (st ^ prev)
            if p is not None: ch = {k for k in ch if abs(k[0] * 6 - p) > 40}
            if ch: ev.append((i / 60, len(ch)))
        prev = st
    m = []
    for t, c in ev:
        if m and t - m[-1][1] < 0.3: m[-1][1] = t; m[-1][2] += c
        else: m.append([t, t, c])
    out[clip] = {"keylit": lo, "edits": [x for x in m if x[2] >= 2]}
    print(f"{clip:7s} runs {[(round(a, 2), round(b, 2), round(D, 3)) for a, b, D, *_ in runs]} | key lights {len(lo)}: median {np.median(d):+.0f} ms, <25 ms {(abs(d) < 25).mean():.2f} | changes {[(round(a, 2), round(b, 2), c) for a, b, c in m if c >= 2][:14]}")
json.dump(out, open("check.json", "w"))
json.dump({"kick": {"t": ons("sub").tolist()}, "snare": {"t": ons("snr").tolist()}, "hat": {"t": ons("hi").tolist()}}, open("ons_master.json", "w"))
