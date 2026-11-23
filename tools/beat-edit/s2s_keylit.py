"""Key-light note-ons (FL lights the key orange at note-on) per row, keyboard x 430-485, grid y 183-938 ->
song times (clock D + 0.02) vs master onsets in a band. usage: s2s_keylit.py clip lo hi"""
import subprocess, json, sys, numpy as np
from scipy.signal import stft, find_peaks
clip, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-vf", "crop=56:756:430:183", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
L = []
while True:
    b = p.stdout.read(56 * 756 * 3)
    if len(b) < 56 * 756 * 3: break
    f = np.frombuffer(b, np.uint8).reshape(756, 56, 3).astype(int)
    o = (f[..., 2] > 200) & (f[..., 2] - f[..., 0] > 90) & (f[..., 1] > 90) & (f[..., 1] < 200)
    L.append(o.mean(1) > 0.4)
L = np.array(L)
C = json.load(open("clock2.json"))[clip]["runs"]
def D(t): return next((r[2] for r in C if r[0] <= t <= r[1]), None)
ev = []
lab = np.zeros(756, int)
for i in range(1, len(L)):
    new = L[i] & ~L[i - 1]
    if new.sum() >= 4:
        rows = np.where(new)[0]
        # group rows into keys
        groups = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
        for g in groups:
            if len(g) >= 4:
                d = D(i / 60)
                if d is not None: ev.append((i / 60 + d + 0.02, int(g.mean()) + 183))
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fq, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); Sx = np.abs(Z); T = np.arange(Sx.shape[1]) * 128 / sr
bb = np.log1p(Sx[(fq >= lo) & (fq < hi)].sum(0) * 10); fl = np.maximum(0, np.diff(bb, prepend=bb[0]))
pk, _ = find_peaks(fl, height=fl.mean() + 0.5 * fl.std(), distance=6); on = T[pk]
BT = 60 / 144
ts = sorted(set(round(t, 3) for t, _ in ev))
dd = np.array([(t - on[np.argmin(abs(on - t))]) * 1000 for t in ts])
grid = np.array([(t / BT * 4 - round(t / BT * 4)) * BT / 4 * 1000 for t in ts])
print(f"{clip}: {len(ts)} key-light note-ons, song {ts[0]:.2f}-{ts[-1]:.2f}s | vs master onsets median {np.median(dd):+.0f} ms, |d|<30 {np.mean(abs(dd) < 30):.2f} | vs 16th grid median {np.median(grid):+.0f} ms, |g|<25 {np.mean(abs(grid) < 25):.2f}")
print("   beats:", [round(t / BT, 2) for t in ts][:40])
json.dump(ev, open(f"{clip}_keylit.json", "w"))
