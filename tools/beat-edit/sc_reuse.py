"""SCOUSE (179 BPM) adapted from FEAR (110 BPM, B minor): coverage per view = (recording, shift in bars). Sections: bars 1-8 and 9-16 = the same
intro/verse loop (808 + chops), bars 17-24 = the drop (nocturne 808, kick, hats, claps, scratch), 25+ = bars 1-8 again.
Window bars 9-20. Clean windows: clock runs with their own offsets; follow rolls (chops, snare, nocturne 808, bass) are
trimmed to frames that show a note near the playhead; clap has notes being placed at 13.6-14.4, 15-16, 17-18 s, snare a
dialog before 2 s, the mixer recording switches to the playlist at ~21 s. Writes cover.json, validate_x.json, hits.json."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 179.0; BAR = 240 / BPM; OF = 0.02; NB = 26
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "40", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
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
FOLLOW_X = {"clap": 357, "clap4": 357, "f808": 862}
def with_notes(rec, wins):
    S = json.load(open(f"{rec}_scan.json")); X = FOLLOW_X[rec]
    L, Rr = ((300, 300) if rec == "f808" else (10, 420))
    ok = np.array([any(X - L < b[0] < X + Rr or (b[0] < X + 5 < b[1]) for b in s_[1] if b[2] > 185) for s_ in S], float)
    ok = np.convolve(ok, np.ones(31) / 31, "same") > 0.15
    out = []
    for a, b, D in wins:
        i0, i1 = int(a * 60), int(b * 60); k = i0
        while k < i1:
            if ok[k]:
                j = k
                while j < i1 and ok[j]: j += 1
                if (j - k) / 60 > 0.5: out.append((k / 60, j / 60, D))
                k = j
            else: k += 1
    return out
REC = {
    "pre": runs("pre", hi=3.06),
    "f808": with_notes("f808", runs("f808")),
    "m808": runs("m808", hi=7.1),
    "clap": with_notes("clap", runs("clap")),
    "hat": runs("hat", lo=6.0, hi=14.85),
    "kick": runs("kick", excl=[(8.4, 9.0)]),        # he draws the bar-14 kick in at clip ~8.7 (pencil)
    "clap4": with_notes("clap4", runs("clap4")),
    "ohat": runs("ohat"),
    "s808": runs("s808"),
    "mixclip": runs("mixclip"),
    "smooth": runs("smooth"),
}
NAME = {"pre": "PRE", "f808": "F808", "m808": "M808", "clap": "CLAP", "hat": "HAT", "kick": "KICK", "s808": "S808", "clap4": "CLAP4", "ohat": "OHAT", "mixclip": "MIXER", "smooth": "SMOOTH"}
FIXED0 = ("pre",)
BCHK = {"f808": ("sub",), "m808": ("sub",), "s808": ("sub",), "kick": ("sub",), "clap": ("snr",), "clap4": ("snr",), "ohat": ("hi",), "hat": ("hi",), "mixclip": ("sub", "mid", "snr", "hi"), "smooth": ("sub", "mid", "snr", "hi")}
VIEWS = {}
for rec in REC:
    for sh in ([0] if rec in FIXED0 else range(-12, 13, 4)):
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
