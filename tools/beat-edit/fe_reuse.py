"""FEAR (110 BPM, B minor): coverage per view = (recording, shift in bars). Sections: bars 1-8 and 9-16 = the same
intro/verse loop (808 + chops), bars 17-24 = the drop (nocturne 808, kick, hats, claps, scratch), 25+ = bars 1-8 again.
Window bars 9-20. Clean windows: clock runs with their own offsets; follow rolls (chops, snare, nocturne 808, bass) are
trimmed to frames that show a note near the playhead; clap has notes being placed at 13.6-14.4, 15-16, 17-18 s, snare a
dialog before 2 s, the mixer recording switches to the playlist at ~21 s. Writes cover.json, validate_x.json, hits.json."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 110.0; BAR = 240 / BPM; OF = 0.02; NB = 26
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
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
FOLLOW_X = {"chops": 736, "snare": 720, "n808": 1152, "bass": 668}
def with_notes(rec, wins):
    """follow rolls: keep only frames with a note between X-130 and X+250 (the 9:16 crop around the playhead)."""
    S = json.load(open(f"{rec}_scan.json")); X = FOLLOW_X[rec]
    ok = np.array([any(X - 130 < b[0] < X + 250 or (b[0] < X < b[1]) for b in s_[1] if b[2] > 195) for s_ in S], float)
    ok = np.convolve(ok, np.ones(31) / 31, "same") > 0.15          # ignore single empty frames; drop >~0.4 s gaps
    out = []
    for a, b, D in wins:
        i0, i1 = int(a * 60), int(b * 60); seg = ok[i0:i1]; k = i0
        while k < i1:
            if ok[k]:
                j = k
                while j < i1 and ok[j]: j += 1
                if (j - k) / 60 > 0.5: out.append((k / 60, j / 60, D))
                k = j
            else: k += 1
    return out
REC = {
    "macro": runs("macro", hi=41.2),
    "chops": with_notes("chops", runs("chops")),
    "snare": with_notes("snare", runs("snare", lo=2.0)),
    "n808": with_notes("n808", runs("n808")),
    "bass": with_notes("bass", runs("bass")),
    "hat": runs("hat", hi=14.4),
    "clap": runs("clap", excl=[(13.5, 14.5), (15.0, 16.05), (17.0, 18.05)], lo=1.2, hi=19.6),
    "mixer": runs("mixer", hi=21.0),
}
NAME = {"macro": "MACRO", "chops": "CHOPS", "snare": "SNARE", "n808": "N808", "bass": "BASS", "hat": "HAT", "clap": "CLAP", "mixer": "MIXER"}
FIXED0 = ("macro",)
BCHK = {"chops": ("mid",), "snare": ("snr",), "n808": ("sub",), "bass": ("sub", "bass"), "hat": ("hi",), "clap": ("snr",),
        "mixer": ("sub", "mid", "snr", "hi")}
VIEWS = {}
for rec in REC:
    for sh in ([0] if rec in FIXED0 else range(-8, 9, 4)):
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
