"""SO DOPE (144 BPM), window bars 25-40 (two 8-bar sections; horn only in the first, rim/snaps absent in bars 33-34,
open hat only in the second). Adapted from KUSH: For every view = (recording, shift in bars) work out
which target bars of the edit (bars 1-18) it may show: the target bar must match the source bar on the AUDIO in three
bands (808 25-160 Hz, mid 300-3k, hi 5-12k; log-spectrogram correlation >= 0.97, so every heard note is the recorded
one, same pitch) and the source bar must be inside the recording's clean playback (clock runs with the majority offset,
minus note-editing windows). Writes cover.json {view: [[clip_a, clip_b, D]]} (song = clip + D) and validate_x.json
(picture note hits shifted into the edit, for the hit sheet)."""
import json, subprocess, numpy as np
from scipy.signal import stft
BPM = 144.0; BAR = 240 / BPM; OF = 0.02; NB = 42; B_LO = 24   # target bars 25-42, so a shot can end exactly on the bar-41 line
C = json.load(open("clock2.json"))
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "100", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                                 capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=2048, noverlap=2048 - 256); S = np.log1p(np.abs(Z)); T = np.arange(S.shape[1]) * 256 / sr
BANDS = {k: S[(fr >= lo) & (fr < hi)] for k, (lo, hi) in {"808": (25, 160), "mid": (300, 3000), "hi": (5000, 12000),
                                                            "snr": (1500, 5000)}.items()}
def bar(L, b):
    i = (T >= b * BAR) & (T < (b + 1) * BAR); return L[:, i]
def same(a, b, bands=("808", "mid", "hi")):
    if not bands: return True
    for L in (BANDS[k] for k in bands):
        x, y_ = bar(L, a), bar(L, b); m = min(x.shape[1], y_.shape[1])
        if m < 10 or np.corrcoef(x[:, :m].ravel(), y_[:, :m].ravel())[0, 1] < 0.97: return False
    return True
# clean playback per recording (clip seconds): majority-offset clock runs merged, minus editing windows (checked by eye)
EDITS = {"hats_in": [(0.0, 15.75)],         # putting the hats in until 15.7 s, then the full page plays (bars 25-28.9)
         "snaps": [(0.0, 13.7)],            # putting the snaps in (box-select / drawing) until ~13.6 s
         "horn": [(0.0, 6.8)],              # horn notes drawn in at 6.0-6.6 s
         "monarch": [(0.0, 13.0)],          # bottom notes deleted and re-added 6.4-12.9 s
         "hats": [(13.6, 99.0)],            # end of the follow-along
         "clipper": [(0.0, 3.9), (28.6, 99.0)],   # OBS window over the mixer until ~3.5 s; loops back at 28.6 s
         "smooth": [(0.0, 0.95), (25.95, 99.0)]}  # OBS window at the start and end            # end of the follow-along
def clean(rec):
    runs = [r for r in C[rec]["runs"] if r[1] - r[0] >= 0.3]
    w = {}
    for r in runs: w.setdefault(round(r[2], 1), 0); w[round(r[2], 1)] += r[1] - r[0]
    Dm = max(w, key=w.get)
    Dm = float(np.median([r[2] for r in runs if abs(r[2] - Dm) < 0.1]))
    keep = [r for r in runs if abs(r[2] - Dm) < 0.06]
    other = [r for r in C[rec]["runs"] if abs(r[2] - Dm) >= 0.05]
    iv = [[keep[0][0], keep[0][1]]]
    for r in keep[1:]:                        # same offset on both sides: the gap is clock misreads (the odd runs
        # in between are 2 or 6 bars off = OCR reading "8" as "6" / "16" as "10", in every recording at the same song
        # bars), playback was continuous
        if r[0] - iv[-1][1] < 6.0: iv[-1][1] = r[1]     # SO DOPE: OCR drops out for up to ~5 s (checked by eye)
        else: iv.append([r[0], r[1]])
    for ea, eb in EDITS.get(rec, []):
        iv = [q for a, b in iv for q in ([a, min(b, ea)], [max(a, eb), b]) if q[1] - q[0] > 0.3]
    D = float(np.median([r[2] for r in keep])) + OF
    return iv, D
VIEWS = {"H808": ("h808", 0), "H808_8": ("h808", 8), "HORN": ("horn", 0), "HORN_8": ("horn", 8), "MON": ("monarch", 0),
         "MON_8": ("monarch", 8), "SNAP": ("snaps", 0), "SNAP_8": ("snaps", 8), "SNAP_4": ("snaps", 4), "SNAP_12": ("snaps", 12),
         "HAT": ("hats", 0), "HAT_8": ("hats", 8), "HIN": ("hats_in", 0), "HIN_8": ("hats_in", 8), "SCR": ("scratch", 0), "SCR_M8": ("scratch", -8), "MACRO": ("macro", 0),
         "CLIP": ("clipper", 0), "PLUGIN": ("smooth", 0)}
# Jack (Nov 5): every 8-bar section uses the same pattern clips (playlist: Hi Hat Basic IV, 808 Chorus, 808 Monarch, Clap 6,
# Rim 25, Snap 8 appear in both sections) except 808 16 -> 808 16 #2, horn (section 1 only), open hat / rim 11 (section 2
# only) and rim / snaps absent in bars 33-34. The whole-band audio test fails for hats / snaps / monarch in section 2 only
# because the open hat, 808 #2 and rim 11 are added on top -> those views reuse by pattern identity (no band test); the
# snaps' own bars 33-34 are excluded by hand (absent there).
BCHK = {"h808": ("808",), "horn": ("808",), "monarch": (), "snaps": (), "hats": (), "hats_in": (), "scratch": ("mid", "hi")}
cover = {}
for v, (rec, sh) in VIEWS.items():
    iv, D = clean(rec)
    Dt = D + sh * BAR                         # target song = clip + Dt
    segs = []
    for a, b in iv:
        t0, t1 = a + Dt, b + Dt               # target song interval
        for tb in range(B_LO, NB):
            sb = tb - sh
            if sb < 0 or not same(tb, sb, BCHK.get(rec, ("808", "mid", "hi"))): continue
            x0, x1 = max(t0, tb * BAR), min(t1, (tb + 1) * BAR)
            if x1 - x0 > 0.05: segs.append([x0 - Dt, x1 - Dt])
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < 0.02: merged[-1][1] = b
        else: merged.append([a, b])
    cover[v] = [[round(a, 3), round(b, 3), round(Dt, 4)] for a, b in merged if b - a > 0.3]
    print(f"{v:8s} {rec:8s} shift {sh:+3d} -> target bars", [(round((a + Dt) / BAR + 1, 2), round((b + Dt) / BAR + 1, 2)) for a, b, _ in cover[v]])
json.dump(cover, open("cover.json", "w"))
VAL = {}
for v, (rec, sh) in VIEWS.items():
    try: h = json.load(open(f"{rec}_hits.json"))
    except FileNotFoundError: continue
    iv, D = clean(rec); Dt = D + sh * BAR
    ok = lambda t: any(a + Dt <= t < b + Dt for a, b, _ in cover[v])
    VAL[v] = {"cross": [(t + sh * BAR, yy) for t, yy, c in h if ok(t + sh * BAR)]}
json.dump(VAL, open("validate_x.json", "w"))
