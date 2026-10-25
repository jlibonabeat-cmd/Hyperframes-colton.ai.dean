"""Per bar: does the recording's own (full-mix) audio match the master at that song time? Spectral-flux correlation
(5 ms hop) per band, best lag within +-30 ms reported."""
import subprocess, numpy as np
SR = 22050; HOP = 110; BAR = 240 / 149; B0 = 0.032
def load(p, ss, t):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{t:.3f}", "-i", p, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
def flux(y, lo, hi):
    w = 1024; n = (len(y) - w) // HOP
    if n < 10: return None
    idx = np.arange(w)[None, :] + HOP * np.arange(n)[:, None]
    X = np.abs(np.fft.rfft(y[idx] * np.hanning(w), axis=1)); F = np.fft.rfftfreq(w, 1 / SR); m = (F >= lo) & (F < hi)
    L = np.log1p(X[:, m] * 20); f = np.concatenate([[0], np.maximum(0, np.diff(L, axis=0)).sum(1)])
    return f - f.mean()
V = {"snare": ("snare.mp4", 11.376, 1.4, 8.6), "hat": ("hat.mp4", 11.719, 1.0, 4.2), "kick": ("kick.mp4", 11.83, 1.0, 13.8),
     "808a": ("s808a.mp4", 10.131, 5.6, 10.6), "808b": ("s808b.mp4", 8.951, 3.8, 16.4), "chops": ("chops.mp4", 11.688, 1.0, 14.6),
     "scratch": ("scratch.mp4", 9.757, 3.0, 8.6), "clipper": ("clipper.mp4", 9.58, 3.2, 27.0), "macro": ("macro.mp4", 4.69, 1.7, 47.3)}
BANDS = {"low": (30, 150), "mid": (150, 2500), "high": (2500, 11000)}
print("bar        " + "".join(f"{b:>7d}" for b in range(4, 24)))
for name, (clip, D, a, b) in V.items():
    for sh in ((0, 8) if name in ("snare", "hat") else (0,)):
        Dp = D + sh * BAR
        row = {k: [] for k in BANDS}
        for bi in range(4, 24):
            s0 = B0 + bi * BAR
            if s0 - Dp < a or s0 + BAR - Dp > b:
                for k in BANDS: row[k].append("     . ")
                continue
            x = load(clip, s0 - Dp - 0.03, BAR + 0.06); y = load("beat.mp3", s0, BAR)
            for k, (lo, hi) in BANDS.items():
                fx, fy = flux(x, lo, hi), flux(y, lo, hi)
                best = max((np.corrcoef(fx[l:l + len(fy)], fy)[0, 1], l) for l in range(0, len(fx) - len(fy)))
                row[k].append(f"{best[0]:5.2f}{(best[1] * 5 - 30):+3.0f}"[:7].rjust(7))
        for k in BANDS:
            print(f"{name}{'+8' if sh else ''}".ljust(8) + f"{k:4s}" + "".join(row[k]))
