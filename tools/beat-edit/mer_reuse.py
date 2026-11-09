"""MERSEY (198 BPM): coverage per view = (recording, whole-bar shift). A target bar is allowed when the source bar is in the
recording's clean playback and, for shifted views, the target bar matches the source bar on the audio (log spectrogram
correlation >= 0.97 in the instrument's bands; snare uses pattern identity: the playlist repeats the same snare clip in
both sections, the extra hits in section 2 are the rims). Writes cover.json {view: [[clip_a, clip_b, D]]} (song = clip + D)
and validate_x.json {view: {"cross": [[song_t, y]]}} (picture note hits shifted into place) and hits.json (master onsets).
Clean windows were found by diffing note pixels (mer_scan.py) and checked by eye: kick / bass / rim recordings are
'putting in' passes, so only the stretches where no note is drawn, dragged or pops in inside the crop are used."""
import json, subprocess, numpy as np
from scipy.signal import stft, find_peaks
BPM = 198.0; BAR = 240 / BPM; OF = 0.02; NB = 27
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "45", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
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
    "macro": [(0.86, 25.6, 8.82)],                     # playback starts at bar 9 at 0.86 s
    "vox": [(0.75, 9.0, 10.003), (9.25, 10.95, 10.02)],
    "snare": [(1.0, 7.7, 10.415)],                      # run 2 (7.8-10.7) scrolls unevenly: not used
    "kick": [(4.0, 8.3, 9.968), (8.85, 10.4, 9.968)],    # kicks drawn at 3.43 s and 8.37-8.8 s
    "bass": [(2.9, 5.5, 10.485), (6.1, 7.9, 10.485), (8.5, 9.35, 10.485)],   # bass drawn / re-drawn at 2.67, 5.65-5.93, 8.1-8.33 s
    "e808": [(14.5, 18.1, 13.083)],                     # the page shows bars 18.5-26.8 (playhead off-page before)
    "perc": [(2.15, 2.75, 18.341)],                     # take 2 of the 'last bar' perc
    "rim_in": [(4.3, 7.3, 16.257), (8.0, 9.0, 16.257), (9.6, 14.6, 16.257)],   # rims drawn ahead of the playhead at 7.43 / 9.13 / 11.22 s
}
VIEWS = {"MACRO": ("macro", 0), "VOX": ("vox", 0), "SNARE": ("snare", 0), "SNARE_8": ("snare", 8),
         "KICK": ("kick", 0), "KICK_8": ("kick", 8), "KICK_M2": ("kick", -2), "KICK_M4": ("kick", -4), "KICK_6": ("kick", 6),
         "BASS": ("bass", 0), "BASS_8": ("bass", 8), "BASS_M2": ("bass", -2), "BASS_2": ("bass", 2), "BASS_6": ("bass", 6), "BASS_10": ("bass", 10),
         "E808": ("e808", 0), "E808_M8": ("e808", -8), "PERC": ("perc", 0), "RIM": ("rim_in", 0)}
BCHK = {"kick": ("sub",), "bass": ("sub", "bass"), "e808": ("sub",), "snare": ()}
cover = {}
for v, (rec, sh) in VIEWS.items():
    segs = []
    for a, b, Dc in REC[rec]:
        D = Dc + OF; Dt = D + sh * BAR
        t0, t1 = a + Dt, b + Dt
        for tb in range(NB):
            sb = tb - sh
            if sb < 0: continue
            if sh and BCHK.get(rec, ("sub", "mid", "hi")) and not same(tb, sb, BCHK[rec]): continue
            x0, x1 = max(t0, tb * BAR), min(t1, (tb + 1) * BAR)
            if x1 - x0 > 0.05: segs.append([x0 - Dt, x1 - Dt, Dt])
    merged = []
    for a, b, Dt in segs:
        if merged and a - merged[-1][1] < 0.02 and abs(Dt - merged[-1][2]) < 1e-6: merged[-1][1] = b
        else: merged.append([a, b, Dt])
    cover[v] = [[round(a, 3), round(b, 3), round(Dt, 4)] for a, b, Dt in merged if b - a > 0.3]
    print(f"{v:8s} -> edit beats", [(round(((a + Dt) / BAR - 8) * 4, 1), round(((b + Dt) / BAR - 8) * 4, 1)) for a, b, Dt in cover[v]])
json.dump(cover, open("cover.json", "w"))
VAL = {}
for v, (rec, sh) in VIEWS.items():
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
