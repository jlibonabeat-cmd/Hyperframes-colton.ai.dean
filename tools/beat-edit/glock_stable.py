"""When does each 'putting in' piano roll show its final pattern? Note sets per frame (10 fps) vs the final state."""
import subprocess, numpy as np, cv2, sys
def notes(f, x):
    m = ((f[..., 1] > 170) & (f[..., 1] - f[..., 2] > 70)).astype(np.uint8)
    m[:180] = 0; m[900:] = 0; m[:, :100] = 0
    if x is not None: m[:, max(0, x - 14):x + 14] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(cx), int(cy)) for (cx, cy), (x0, y0, w, h, a) in zip(_[1:], st[1:]) if a > 25] if False else \
           [(int(x0 + w / 2), int(y0 + h / 2)) for x0, y0, w, h, a in st[1:] if a > 25]
def ph(f):
    reg = f[200:880:4]; g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5); c = g.mean(0)
    xs = np.where(c[100:] > 0.6)[0]; return int(xs[0]) + 100 if len(xs) else None
for clip, a, b in [("snare.mp4", 1.4, 8.6), ("kick.mp4", 1.0, 13.8), ("s808a.mp4", 5.6, 10.6), ("scratch.mp4", 3.0, 8.6), ("hat.mp4", 1.0, 4.2)]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", clip, "-vf", "fps=10", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    S = []
    while True:
        buf = p.stdout.read(1080 * 1920 * 3)
        if len(buf) < 1080 * 1920 * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); x = ph(f)
        S.append((x, notes(f, x)))
    # final = union over the last 1 s (playhead masks a strip each frame)
    fin = []
    for x, ns in S[-10:]:
        for n in ns:
            if not any(abs(n[0] - q[0]) <= 6 and abs(n[1] - q[1]) <= 6 for q in fin): fin.append(n)
    row = []
    for k, (x, ns) in enumerate(S):
        miss = [q for q in fin if (x is None or abs(q[0] - x) > 16) and not any(abs(n[0] - q[0]) <= 6 and abs(n[1] - q[1]) <= 6 for n in ns)]
        extra = [n for n in ns if not any(abs(n[0] - q[0]) <= 6 and abs(n[1] - q[1]) <= 6 for q in fin)]
        row.append("." if not miss and not extra else ("m" if miss and not extra else ("e" if extra and not miss else "x")))
    print(f"{clip:12s} {a:4.1f}s |" + "".join(row) + f"| {b:.1f}s   final notes {len(fin)}")
    for k in range(0, len(S), 10):
        x, ns = S[k]
        miss = [q for q in fin if (x is None or abs(q[0] - x) > 16) and not any(abs(n[0] - q[0]) <= 6 and abs(n[1] - q[1]) <= 6 for n in ns)]
        if miss: print("     t", round(a + k / 10, 1), "missing", miss[:6])
