"""Where is the play line really? At every master onset in the element's band (inside the recording's clean runs) take
the picture frame and histogram the note left edges (follow rolls) or note-edge minus FL playhead (page rolls). The peak
is the play position; its spread is the timing jitter. usage: fe_onsetx.py clip band [page]"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band = sys.argv[1], sys.argv[2]; page = len(sys.argv) > 3
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 1.5 * fl.std(), distance=10); on = T[pk]
rel, on_bg = [], []
for s in on:
    for a, b, D, *_ in C:
        t = s - D - 0.02
        if a + 0.1 <= t <= b - 0.1:
            i = int(round(t * 60))
            p, bx = S[i]
            for x0, x1, yy in bx:
                if yy < 195 or x0 < 575: continue
                if page:
                    if p is not None: rel.append(x0 - p)
                else: rel.append(x0)
            # background: frame half a 16th away
            j = i + 4
            if j < len(S):
                for x0, x1, yy in S[j][1]:
                    if yy >= 195 and x0 >= 575: on_bg.append(x0 - (S[j][0] or 0) if page else x0)
rel = np.array(rel); bg = np.array(on_bg)
rng = (-60, 60) if page else (575, 1880)
h, e = np.histogram(rel, bins=np.arange(*rng, 2)); hb, _ = np.histogram(bg, bins=e)
sc = h - 0.8 * hb; k = int(np.argmax(sc)); X = e[k] + 1
near = rel[np.abs(rel - X) < 20]
print(f"{clip}: {len(on)} onsets, peak at {'rel ' if page else 'x='}{X:.0f} (count {h[k]} vs bg {hb[k]}); spread of edges within 20 px: sd {near.std():.1f}px")
top = np.argsort(-sc)[:6]; print("  top bins", [(int(e[i] + 1), int(h[i]), int(hb[i])) for i in top])
