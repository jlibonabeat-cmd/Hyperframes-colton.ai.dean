"""Page-mode note hits (MERSEY, brown theme). The notes sit still on a page while the playhead sweeps it, so a hit is
the moment the playhead x crosses a note's left edge. Pages = runs where the playhead moves right; notes = boxes that
stay put in >= 30% of a page's frames. song = clip + D (clock run) + OF. Compares to master onsets in a band.
usage: mer_hits.py clip band [OF=0.02]   -> {clip}_hits.json [(song_t, y, clip_t)]"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band = sys.argv[1], sys.argv[2]; OF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
FPS = 60; BAR = 240 / 193
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]
ph = np.array([np.nan if s[0] is None else s[0] for s in S])
def Dp(t):
    for a, b, D, *_ in C:
        if a - 0.1 <= t <= b + 0.1: return D
    return None
# pages: playhead moving right; a backwards jump > 150 px starts a new page
pages, cur = [], []
for i, x in enumerate(ph):
    if np.isnan(x): continue
    if cur and (x < ph[cur[-1]] - 150 or i - cur[-1] > 20): pages.append(cur); cur = []
    cur.append(i)
if cur: pages.append(cur)
hits = []
for pg in pages:
    if len(pg) < 15: continue
    cnt = {}
    for i in pg:
        for a, b, y in S[i][1]:
            if y < 235 or a <= 476: continue
            k = (int(round(a / 4)), int(round(y / 6)))
            cnt[k] = cnt.get(k, 0) + 1
    xs = sorted({k for k, c in cnt.items() if c >= 0.3 * len(pg)})
    fr = np.array(pg, float); px = ph[pg]
    for kx, ky in xs:
        x0 = kx * 4
        if not (px.min() < x0 < px.max()): continue
        j = np.searchsorted(px, x0)
        if j == 0 or j >= len(px): continue
        f = fr[j - 1] + (x0 - px[j - 1]) / max(px[j] - px[j - 1], 1e-6) * (fr[j] - fr[j - 1])
        t = f / FPS; D = Dp(t)
        if D is not None: hits.append((round(t + D + OF, 3), ky * 6, round(t, 3)))
hits = sorted(set(hits))
# merge duplicates (same note seen twice, keys 4 px apart)
out = []
for h in hits:
    if out and abs(h[0] - out[-1][0]) < 0.03 and abs(h[1] - out[-1][1]) <= 6: continue
    out.append(h)
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "45", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
b = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(b, prepend=b[0]))
pk, _ = find_peaks(fl, height=fl.mean() + fl.std(), distance=8); on = T[pk]
d = [(h[0] - on[np.argmin(abs(on - h[0]))]) * 1000 for h in out]
print(clip, len(out), "hits; median offset vs", band, "onsets", round(float(np.median(d)), 1) if d else None, "ms; |d|<40ms:", sum(abs(x) < 40 for x in d))
for h, dd in zip(out, d):
    print(f"  song {h[0]:7.3f} bar {h[0] / BAR + 1:6.2f} y{h[1]:4d} clip {h[2]:6.3f}  {dd:+6.0f} ms")
json.dump(out, open(f"{clip}_hits.json", "w"))
