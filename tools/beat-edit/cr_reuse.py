"""CREEPIN THROUGH THE MANSION (193 BPM, B minor): coverage per view = (recording, shift in bars, multiples of 4).
Sections: bar 1 intro (melody only), 8-bar sections bars 2-9, 10-17, 18-25 (bars 9/17/25 = section end: counter snare,
OK vox at 25). A shifted target bar is allowed only if it matches the source bar on the audio (log spectrogram >= 0.97
in the element's bands; mixer/plugins all bands). Clean windows: 'snare recording' is him drawing snares in until 13.4 s,
the 808 pattern has notes dragged until 5.2 s, clap has the cursor over the notes at 5.1-5.45 / 8.2-8.45 s.
Writes cover.json {view: [[clip_a, clip_b, D]]}, validate_x.json, hits.json."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 193.0; BAR = 240 / BPM; OF = 0.02; NB = 26
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                                 capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=2048, noverlap=2048 - 256); S = np.log1p(np.abs(Z)); T = np.arange(S.shape[1]) * 256 / sr
BANDS = {k: S[(fr >= lo) & (fr < hi)] for k, (lo, hi) in {"sub": (25, 100), "bass": (100, 300), "mid": (300, 3000),
                                                            "hi": (5000, 12000), "snr": (1500, 5000)}.items()}
def bar(L, b): i = (T >= b * BAR) & (T < (b + 1) * BAR); return L[:, i]
def same(a, b, bands):
    for k in bands:
        x, y_ = bar(BANDS[k], a), bar(BANDS[k], b); m = min(x.shape[1], y_.shape[1])
        if m < 10 or np.corrcoef(x[:, :m].ravel(), y_[:, :m].ravel())[0, 1] < 0.97: return False
    return True
# (recording, clean clip windows with their clock offset D (FL time = clip + D))
REC = {
    "macro": [(2.1, 33.2, -2.064)],
    "melody": [(4.35, 8.95, -3.101), (9.2, 12.6, -3.098), (13.05, 14.25, -3.098)],
    "tweak": [(0.3, 7.25, 11.606), (8.0, 10.7, 11.607)],
    "clap": [(3.85, 5.1, -2.645), (5.45, 7.95, -2.645), (8.85, 12.45, -2.64)],
    "snare": [(13.45, 17.25, -6.366)],
    "csnare": [(1.4, 3.0, 8.507)],
    "hat": [],   # a hat pair every ~2 bars; picture crossings don't line up with the master hats -> not used
    "m808": [(5.25, 7.9, 8.275)],
    "s808": [(0.9, 4.4, 10.299), (4.45, 6.0, 10.292), (6.05, 9.35, 10.298)],   # after 9.5 s the page runs past the pattern end
    "vox": [(1.55, 3.1, 28.29)],
    "djfx": [(4.15, 7.4, 10.71)],
    "smooth": [(1.05, 4.9, -1.053), (5.0, 6.0, -1.053), (6.45, 12.25, -1.059)],
    "clipper": [(0.65, 5.8, 13.322), (6.45, 13.0, 13.322)],
    "mixer": [(0.3, 8.9, 34.075), (9.0, 14.45, 34.077)],
}
NAME = {"macro": "MACRO", "melody": "MEL", "tweak": "TWEAK", "clap": "CLAP", "snare": "SNARE", "csnare": "CSN", "hat": "HAT",
        "m808": "M808", "s808": "S808", "vox": "VOX", "djfx": "DJFX", "smooth": "PLUGIN", "clipper": "CLIP", "mixer": "MIXER"}
FIXED0 = ("macro", "tweak", "csnare", "vox", "djfx")     # shown only where they really play
BCHK = {"melody": ("mid",), "clap": ("snr",), "snare": ("snr",), "hat": ("hi",), "m808": ("sub",), "s808": ("sub",),
        "smooth": ("sub", "mid", "snr", "hi"), "clipper": ("sub", "mid", "snr", "hi"), "mixer": ("sub", "mid", "snr", "hi")}
VIEWS = {}
for rec in REC:
    for sh in ([0] if rec in FIXED0 else range(-32, 33, 4)):
        VIEWS[NAME[rec] + ("" if sh == 0 else f"_{'m' if sh < 0 else ''}{abs(sh)}")] = (rec, sh)
cover = {}
for v, (rec, sh) in VIEWS.items():
    segs = []
    for a, b, Dc in REC[rec]:
        D = Dc + OF; Dt = D + sh * BAR
        t0, t1 = a + Dt, b + Dt
        for tb in range(NB):
            sb = tb - sh
            if sb < 0: continue
            if sh and not same(tb, sb, BCHK[rec]): continue
            x0, x1 = max(t0, tb * BAR), min(t1, (tb + 1) * BAR)
            if x1 - x0 > 0.05: segs.append([x0 - Dt, x1 - Dt, Dt])
    merged = []
    for a, b, Dt in segs:
        if merged and a - merged[-1][1] < 0.02 and abs(Dt - merged[-1][2]) < 1e-6: merged[-1][1] = b
        else: merged.append([a, b, Dt])
    cover[v] = [[round(a, 3), round(b, 3), round(Dt, 4)] for a, b, Dt in merged if b - a > 0.3]
    if cover[v]: print(f"{v:10s} -> beats", [(round((a + Dt) / BAR * 4, 1), round((b + Dt) / BAR * 4, 1)) for a, b, Dt in cover[v]])
cover = {k: v for k, v in cover.items() if v}
json.dump(cover, open("cover.json", "w"))
VAL = {}
for v, (rec, sh) in VIEWS.items():
    if v not in cover: continue
    try: h = json.load(open(f"{rec}_hits.json"))
    except FileNotFoundError: continue
    ok = lambda t: any(a + Dt <= t < b + Dt for a, b, Dt in cover[v])
    VAL[v] = {"cross": [(t + sh * BAR, yy) for t, yy, c in h if ok(t + sh * BAR)]}
json.dump(VAL, open("validate_x.json", "w"))
# master onsets for the builder's activity scores
sx_fr, _, Zx = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Zx); Tx = np.arange(Sx.shape[1]) * 128 / sr
def ons(lo, hi):
    bb = np.log1p(Sx[(sx_fr >= lo) & (sx_fr < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
    pk, pr = find_peaks(fl, height=fl.mean() + fl.std(), distance=8)
    s = pr["peak_heights"] / pr["peak_heights"].max()
    return {"t": Tx[pk].tolist(), "s": s.tolist()}
json.dump({"kick": ons(30, 100), "snare": ons(1500, 5000), "hat": ons(5000, 11000)}, open("hits.json", "w"))
