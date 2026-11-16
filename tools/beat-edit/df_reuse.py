"""DRUG FIT (181 BPM, C minor): coverage per view = (recording, shift in bars), checked on the master's audio.
Master: bars 1-8 = intro loop (808 on the odd bars, snares, hats), bars 9-16 = the 808 section (bar 16 no 808), 17-32 the
same again. Recordings: bars 1-8 (snare follow, snare fill, claps, CTM 808, open hats, spinz 808), bars 9-16 (808 main),
mixer bars 1-17, macro ~0.85 s of playback at bar 1. Several are "putting in" recordings: every note edit found by
df_editpx.py is cut out, and so is every note hit the MASTER doesn't have (the project has claps at beats 7.07 / 23.08
that are not in the mp3; the snare fill has a note at beat 13.97 the mp3 doesn't play) - see df_cross.py checks.
Writes cover.json, validate_x.json, hits.json."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 181.0; BAR = 240 / BPM; OF = 0.02; NB = 17
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "30", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
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
# note-edit windows from df_editpx.py (pixel-exact, +-0.12 s) and hits the mp3 doesn't play (df_cross.py)
REC = {
    "macro": runs("macro"),                                       # 3.35-4.2: the only playback in the macro recording
    "snare": runs("snare"),
    "snfill": runs("snfill", excl=[(2.2, 2.55), (3.6, 3.9), (6.4, 6.7)]),
    "clap": runs("clap", hi=5.35),                                # claps drawn in from 5.43 on; beat-7/23 claps not in the mp3
    "ctm808": runs("ctm808", excl=[(3.6, 4.1), (8.6, 8.9)]),
    "ohat": runs("ohat", excl=[(3.55, 3.85), (6.15, 6.45), (8.9, 9.2), (11.35, 11.7)]),   # an open hat drawn in every ~2.6 s
    "spinz": runs("spinz", excl=[(4.85, 5.2), (7.2, 7.55), (9.8, 10.1), (12.15, 12.5)]),
    "m808": runs("m808", excl=[(6.7, 7.2), (7.7, 8.2), (10.3, 10.9), (11.8, 12.3), (12.9, 13.4), (13.65, 14.45)]),
    "mixer": runs("mixer"),
}
NAME = {"macro": "MACRO", "snare": "SNARE", "snfill": "SNFILL", "clap": "CLAP", "ctm808": "CTM", "ohat": "OHAT", "spinz": "SPINZ", "m808": "M808", "mixer": "MIXER"}
FIXED0 = ("macro",)
BCHK = {"snare": ("snr", "hi"), "snfill": ("snr",), "clap": ("snr",), "ohat": ("hi",), "ctm808": ("sub",), "spinz": ("sub",), "m808": ("sub",), "mixer": ("sub", "mid", "snr", "hi")}
VIEWS = {}
for rec in REC:
    for sh in ([0] if rec in FIXED0 else (-8, -4, 0, 4, 8)):
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
