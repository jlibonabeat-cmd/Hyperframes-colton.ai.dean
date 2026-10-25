"""Per recording: every playhead/note crossing (song time, note y) + crossings of final-pattern notes that are NOT yet on
screen ('missing' = heard but not shown). Writes validate.json for the shot planner/renderer."""
import subprocess, json, numpy as np, cv2
from scipy.signal import butter, sosfilt, find_peaks
SR = 22050
V = {"SNARE": ("snare.mp4", 11.376, 1.4, 8.6, (1500, 5000)), "KICK": ("kick.mp4", 11.83, 1.0, 13.8, (35, 150)),
     "808A": ("s808a.mp4", 10.131, 5.6, 10.6, (30, 150)), "SCRATCH": ("scratch.mp4", 9.757, 3.0, 8.6, (300, 6000)),
     "CHOPS": ("chops.mp4", 11.688, 1.0, 14.6, (250, 3000)), "HAT": ("hat.mp4", 11.719, 1.0, 4.2, (7000, 10500))}
PIC = 0.02
def boxes(f, x):
    m = ((f[..., 1] > 170) & (f[..., 1] - f[..., 2] > 70)).astype(np.uint8)
    m[:180] = 0; m[905:] = 0; m[:, :100] = 0
    if x is not None: m[:, max(0, x - 14):x + 14] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x0), int(y0 + h / 2), int(w)) for x0, y0, w, h, a in st[1:] if a > 25 and h >= 6]
def ph(f):
    reg = f[200:880:4]; g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5); c = g.mean(0)
    xs = np.where(c[100:] > 0.6)[0]; return int(xs[0]) + 100 if len(xs) else None
def onsets(lo, hi):
    y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "beat.mp3", "-t", "60", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
    e = np.abs(sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y)); e = np.convolve(e, np.ones(55) / 55, mode="same")[::55]
    le = np.log1p(e * 300); fl = np.maximum(0, np.diff(le, prepend=le[0])); pk, _ = find_peaks(fl, height=np.percentile(fl, 90), distance=20)
    return (pk * 55 / SR).tolist()
out = {}
for name, (clip, D, a, b, band) in V.items():
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", clip, "-vf", "fps=30", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    S = []
    while True:
        buf = p.stdout.read(1080 * 1920 * 3)
        if len(buf) < 1080 * 1920 * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); x = ph(f); S.append((x, boxes(f, x)))
    fin = []
    for x, bs in S[-30:]:
        for q in bs:
            if not any(abs(q[0] - r[0]) <= 6 and abs(q[1] - r[1]) <= 6 for r in fin): fin.append(q)
    cross = []; miss = []
    for k in range(16, len(S)):
        x, x1, xo = S[k][0], S[k - 1][0], S[k - 15][0]
        if x is None or x1 is None or xo is None or not (0 < x - x1 < 30) or not (0 < x - xo < 300): continue
        t = a + (k - 0.5) / 30 + D + PIC          # song time (OBS picture leads its audio ~20 ms)
        shown = S[k - 15][1]
        for q in shown:
            if x1 < q[0] <= x: cross.append((round(t, 3), q[1]))
        for q in fin:
            if x1 < q[0] <= x and not any(abs(q[0] - r[0]) <= 6 and abs(q[1] - r[1]) <= 6 for r in shown): miss.append((round(t, 3), q[1]))
    on = np.array(onsets(*band))
    prec = [(t, y, round(1000 * float(np.min(np.abs(on - t))))) for t, y in cross]
    out[name] = {"D": D, "a": a, "b": b, "cross": prec, "miss": miss}
    bad = [c for c in prec if c[2] > 50]
    print(f"{name:8s} crossings {len(prec):3d}  off-beat(>50ms) {len(bad):2d}  missing-note crossings {len(miss):2d}  final notes {len(fin)}")
    print("   off:", [(c[0], c[2]) for c in bad][:12]); print("   miss:", miss[:12])
json.dump(out, open("validate.json", "w"))
