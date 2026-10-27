"""Light-theme FL piano roll scan at native 60 fps: per frame (playhead_x, follow_flag, note boxes). <clip>_scan.json
Playhead core colour ~BGR(92,148,191) (page view: moving column); follow views keep it at the keyboard edge x=352."""
import subprocess, json, sys, numpy as np, cv2
def core(f):
    return (np.abs(f[..., 0] - 91) < 22) & (np.abs(f[..., 1] - 140) < 30) & (np.abs(f[..., 2] - 185) < 28)
def notemask(f):
    return (f[..., 2] > 200) & (f[..., 0] < 115) & (f[..., 1] > 70) & (f[..., 1] < 180)
def ph(f):
    c = core(f[210:870:5]).mean(0); xs = np.where(c[358:1900] > 0.8)[0]
    return int(xs[0]) + 358 if len(xs) else None
def boxes(f, x):
    m = notemask(f).astype(np.uint8)
    m[:205] = 0; m[878:] = 0; m[:, :353] = 0
    if x is not None: m[:, max(0, x - 3):x + 3] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return [(int(x0), int(y0 + h / 2), int(w)) for x0, y0, w, h, a in st[1:] if a > 30 and h >= 8]
clip = sys.argv[1]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
S = []
while True:
    buf = p.stdout.read(1080 * 1920 * 3)
    if len(buf) < 1080 * 1920 * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).astype(np.int16); x = ph(f); S.append((x, boxes(f, x)))
json.dump(S, open(clip + "_scan.json", "w")); print(clip, len(S), "frames; moving playhead in", sum(s[0] is not None for s in S))
