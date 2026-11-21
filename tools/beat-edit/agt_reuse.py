"""AINT GOT TIME (140 BPM, G minor sample): coverage per view = (recording, shift in bars), checked on the master.
Arrangement (macro, bar 1 = x 342, 54.5 px a bar): vocal chop bars 1-16, Pigments #3 bass bars 1-16, steel-drum accent
bars 1-7, flute bars 5-9 (and 20-24), Pigments chords + Pigments #4 counter bars 9-24, Serum FX bars 17-20; bars 25+ tail.
The OBS recordings have no audio: sync is FL's clock (+20 ms picture lead), checked on the picture against the master
(agt_sync.py: bass +2 ms, counter +6 ms, chords +11 ms). Note edits (agt_editpx.py) cut out: bass F3 drawn at clip 7.4,
counter notes drawn at 13.8 / 14.4. Writes cover.json, validate_x.json, hits.json."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 140.0; BAR = 240 / BPM; OF = 0.02; NB = 16
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "48", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                                 capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=2048, noverlap=2048 - 256); S = np.log1p(np.abs(Z)); T = np.arange(S.shape[1]) * 256 / sr
BANDS = {k: S[(fr >= lo) & (fr < hi)] for k, (lo, hi) in {"sub": (25, 100), "bass": (60, 250), "mid": (300, 3000),
                                                            "hi": (5000, 12000), "snr": (1500, 5000)}.items()}
def bar(L, b): i = (T >= b * BAR) & (T < (b + 1) * BAR); return L[:, i]
def same(a, b, bands):
    for k in bands:
        x, y_ = bar(BANDS[k], a), bar(BANDS[k], b); m = min(x.shape[1], y_.shape[1])
        if m < 10 or np.corrcoef(x[:, :m].ravel(), y_[:, :m].ravel())[0, 1] < 0.97: return False
    return True
# (recording, clean clip windows with their clock offset D (FL time = clip + D))
CLK = json.load(open("clock2.json"))
def runs(rec, excl=(), lo=0.0, hi=999.0):
    out = []; rr = []
    for a, b, D, *_ in CLK[rec]["runs"]:            # same offset on both sides of a short OCR gap: one continuous run
        if rr and abs(D - rr[-1][2]) < 0.03 and a - rr[-1][1] < 0.6: rr[-1] = [rr[-1][0], b, rr[-1][2]]
        else: rr.append([a, b, D])
    for a, b, D in rr:
        a, b = max(a, lo), min(b, hi)
        iv = [[a, b]]
        for ea, eb in excl:
            iv = [q for x, y in iv for q in ([x, min(y, ea)], [max(x, eb), y]) if q[1] - q[0] > 0.3]
        out += [(x, y, D) for x, y in iv if y - x > 0.5]
    return out
REC = {
    "macro": runs("macro"),
    "flute": runs("flute"),
    "bass": runs("bass", excl=[(7.2, 7.6)], hi=13.5),             # F3 drawn in at 7.4; page scrolls on from 13.5
    "counter": runs("counter", excl=[(13.7, 14.55)], lo=12.0),     # notes drawn in at 13.8 / 14.4; playhead on page from 12.0
    "chords": runs("chords", hi=17.3),                            # pattern end scrolls into view after 17.3
    "mixer": runs("mixer", excl=[(12.4, 29.5)]),                   # mixer re-laid out at 12.5 (browser closed), OBS popup
    "shaper": runs("shaper", excl=[(3.9, 4.6), (10.9, 11.5)]),     # LFO shape changed at ~4.3 and ~11.2
}
NAME = {"macro": "MACRO", "flute": "FLUTE", "bass": "BASS", "counter": "COUNTER", "chords": "CHORDS", "mixer": "MIXER", "shaper": "SHAPER"}
FIXED0 = ("macro", "flute", "mixer", "shaper", "chords")
BCHK = {"bass": (), "counter": ("mid",)}   # bass: the same Pigments #3 pattern clip restarts at bar 9 (pattern identity)
VIEWS = {}
for rec in REC:
    for sh in ([0] if rec in FIXED0 else (0, 8)):
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
