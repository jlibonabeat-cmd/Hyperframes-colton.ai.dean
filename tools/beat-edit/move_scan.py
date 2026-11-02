"""Purple FL theme piano rolls (MOVE): per native frame -> (playhead_x, [note boxes x0,x1,yc]).
Playhead and notes share the same pink-purple (R~178 G~85 B~220); the playhead is the one column that covers
most rows of the grid, notes are short horizontal boxes. usage: move_scan.py clip [y0 y1 x0]"""
import subprocess, json, sys, numpy as np, cv2
clip = sys.argv[1]
Y0, Y1, X0 = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (190, 885, 456)
def mask(f):
    R, G, B = f[..., 2], f[..., 1], f[..., 0]
    return (B > 190) & (R > 150) & (G > 55) & (G < 130)
def ph(m):
    cov = m[::3].mean(0)
    xs = np.where(cov > 0.55)[0]
    return float(np.median(xs)) if len(xs) else None
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(1080 * 1920 * 3)
    if len(buf) < 1080 * 1920 * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)[Y0:Y1, X0:1885].astype(np.int16)
    m = mask(f); x = ph(m)
    mm = m.astype(np.uint8)
    if x is not None: mm[:, max(0, int(x) - 3):int(x) + 4] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
    bx = [(int(a) + X0, int(a + w) + X0, int(b + h / 2) + Y0) for a, b, w, h, ar in st[1:] if 8 <= h <= 24 and w >= 3 and ar > 24]
    S.append((None if x is None else x + X0, bx))
json.dump(S, open(clip + "_scan.json", "w"))
print(clip, len(S), "frames; playhead in", sum(s[0] is not None for s in S), "| mid boxes", S[len(S) // 2])
