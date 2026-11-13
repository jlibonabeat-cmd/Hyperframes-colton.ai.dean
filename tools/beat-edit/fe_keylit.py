"""FL lights the piano-roll key orange while its note plays: a ground-truth 'note on' clock for follow rolls whose
playhead is hidden. Find key-light onsets per row, map to song time (clock), compare with master onsets in the band.
usage: fe_keylit.py clip band [kx0 kx1]"""
import json, sys, subprocess, numpy as np
from scipy.signal import stft, find_peaks
clip, band = sys.argv[1], sys.argv[2]
KX0, KX1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (525, 552)
C = json.load(open("clock2.json"))[clip]["runs"]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-vf", f"crop={KX1-KX0}:745:{KX0}:190", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
W = KX1 - KX0; lit = []
while True:
    buf = p.stdout.read(W * 745 * 3)
    if len(buf) < W * 745 * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(745, W, 3).astype(int)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    m = ((R > 200) & (G > 130) & (G < 215) & (B < 170)).mean(1) > 0.5      # orange key rows
    lit.append(m)
lit = np.array(lit)
rows = lit.any(0); ons = []
for i in range(1, len(lit)):
    new = lit[i] & ~lit[i - 1]
    if new.sum() >= 4:
        t = i / 60; D = next((r[2] for r in C if r[0] <= t <= r[1]), None)
        if D is not None: ons.append(t + D + 0.02)
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "70", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
lo, hi = {"sub": (30, 100), "low": (100, 400), "mid": (300, 3000), "hi": (5000, 11000), "snr": (1500, 5000)}[band]
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0)); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 0.8 * fl.std(), distance=8); on = T[pk]
d = np.array([(t - on[np.argmin(abs(on - t))]) * 1000 for t in ons])
print(f"{clip}: {len(ons)} key-light onsets; vs master {band}: median {np.median(d):+.0f} ms, within 25 ms {(abs(d) < 25).mean():.2f}, within 50 {(abs(d) < 50).mean():.2f}")
json.dump(ons, open(f"{clip}_keylit.json", "w"))
