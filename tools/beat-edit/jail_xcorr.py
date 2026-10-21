import subprocess, json, sys, numpy as np
SR = 11025; HOP = 110  # ~10ms
def load(path, ss=None, t=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(ss)] if ss is not None else []) + ["-i", path] + (["-t", str(t)] if t else []) + ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)
def onset(y):
    n = len(y) // HOP
    fr = y[:n * HOP].reshape(n, HOP)
    import numpy.fft as F
    # simple spectral flux with 512 window
    w = 512; X = []
    pad = np.concatenate([y, np.zeros(w)])
    for i in range(n):
        seg = pad[i * HOP:i * HOP + w] * np.hanning(w)
        X.append(np.abs(np.fft.rfft(seg)))
    X = np.log1p(np.array(X))
    fl = np.maximum(0, np.diff(X, axis=0)).sum(1)
    fl = np.concatenate([[0], fl])
    fl = fl - np.convolve(fl, np.ones(50) / 50, mode="same")
    return np.maximum(fl, 0)
beat = onset(load("beat.mp3", 0, 80))
cm = json.load(open("clockmap.json"))
res = {}
for clip in sys.argv[1:]:
    y = onset(load(f"{clip}.mp4"))
    out = []
    for (a, b, D) in cm[clip + "_runs"]:
        i0, i1 = int(a * 100), int(b * 100)
        seg = y[i0:i1]
        if len(seg) < 100 or seg.std() == 0:
            continue
        best = None
        for lag in range(-40, 41):          # +-0.4 s around clock D
            j0 = int(round((a + D) * 100)) + lag
            ref = beat[j0:j0 + len(seg)]
            if len(ref) < len(seg): continue
            c = np.corrcoef(seg, ref)[0, 1]
            if best is None or c > best[0]: best = (c, lag)
        out.append((a, b, round(D, 3), round(D + best[1] / 100, 3), round(best[0], 3)))
    print(clip, out)
    res[clip] = out
json.dump(res, open("xcorr.json", "w"))
