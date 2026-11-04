"""KUSH (193 BPM): the beat is an 8-bar loop (bar 8 / 16 / 17 vary). For every view = (recording, shift in bars) work out
which target bars of the edit (bars 1-18) it may show: the target bar must match the source bar on the AUDIO in three
bands (808 25-160 Hz, mid 300-3k, hi 5-12k; log-spectrogram correlation >= 0.97, so every heard note is the recorded
one, same pitch) and the source bar must be inside the recording's clean playback (clock runs with the majority offset,
minus note-editing windows). Writes cover.json {view: [[clip_a, clip_b, D]]} (song = clip + D) and validate_x.json
(picture note hits shifted into the edit, for the hit sheet)."""
import json, subprocess, numpy as np
from scipy.signal import stft
BPM = 193.0; BAR = 240 / BPM; OF = 0.02; NB = 19   # bar 19 too, so a shot can end exactly on the bar-19 line
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
    for L in (BANDS[k] for k in bands):
        x, y_ = bar(L, a), bar(L, b); m = min(x.shape[1], y_.shape[1])
        if m < 10 or np.corrcoef(x[:, :m].ravel(), y_[:, :m].ravel())[0, 1] < 0.97: return False
    return True
# clean playback per recording (clip seconds): majority-offset clock runs merged, minus editing windows (checked by eye)
EDITS = {"s808_in": [(0.0, 9.95)],          # second 808 is being drawn until clip ~9.9 s
         "m808f": [(10.15, 11.9)]}          # follow-along runs past the pattern end (empty grey page) before looping
def clean(rec):
    runs = [r for r in C[rec]["runs"] if r[1] - r[0] >= 0.3]
    w = {}
    for r in runs: w.setdefault(round(r[2], 1), 0); w[round(r[2], 1)] += r[1] - r[0]
    Dm = max(w, key=w.get)
    keep = [r for r in runs if abs(r[2] - Dm) < 0.05]
    other = [r for r in C[rec]["runs"] if abs(r[2] - Dm) >= 0.05]
    iv = [[keep[0][0], keep[0][1]]]
    for r in keep[1:]:                        # same offset on both sides: the gap is clock misreads (the odd runs
        # in between are 2 or 6 bars off = OCR reading "8" as "6" / "16" as "10", in every recording at the same song
        # bars), playback was continuous
        if r[0] - iv[-1][1] < 2.5: iv[-1][1] = r[1]
        else: iv.append([r[0], r[1]])
    for ea, eb in EDITS.get(rec, []):
        iv = [q for a, b in iv for q in ([a, min(b, ea)], [max(a, eb), b]) if q[1] - q[0] > 0.3]
    D = float(np.median([r[2] for r in keep])) + OF
    return iv, D
VIEWS = {"M808": ("m808f", 0), "M808_8": ("m808f", 8), "S808": ("s808_in", 0), "S808_M8": ("s808_in", -8),
         "SNARE": ("snare", -24), "SNARE_32": ("snare", -32), "PERC": ("perc", -16), "PERC_24": ("perc", -24),
         "OH": ("oh", -16), "OH_24": ("oh", -24), "PERC2": ("perc2", -16), "PERC2_24": ("perc2", -24),
         "MIXER": ("mixer", 0), "PLUGIN": ("smooth", 0), "MACRO": ("macro", 0), "CLIP": ("clipper", -56),
         "CLIP_48": ("clipper", -48), "CLIP_40": ("clipper", -40), "M808_16": ("m808f", 16), "MIXER_8": ("mixer", 8),
         "PLUGIN_8": ("smooth", 8), "S808_8": ("s808_in", 8)}
# bands that must repeat for each recording: 808 rolls -> 808 band; drum rolls -> 808 + hi (the drums; the melody in the
# mid band changes between cycles and doesn't matter for a drum shot); mixer / plugins / arrangement -> everything
BCHK = {"m808f": ("808",), "s808_in": ("808",), "snare": ("808", "hi", "snr"), "perc": ("808", "hi"), "oh": ("808", "hi"),
        "perc2": ("808", "hi")}
cover = {}
for v, (rec, sh) in VIEWS.items():
    iv, D = clean(rec)
    Dt = D + sh * BAR                         # target song = clip + Dt
    segs = []
    for a, b in iv:
        t0, t1 = a + Dt, b + Dt               # target song interval
        for tb in range(NB):
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
