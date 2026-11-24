"""AIN'T GOT TIME - SEGMENTED variation (after @kailash.beats): face panel on top, stacked strips of the recordings below,
all in sync. Bars 1-8: bass page + arrangement (2 strips). The drop (bar 9): warm light flash, 3 strips: ShaperBox ->
chords roll (from beat 37, when the re-recorded chords start) / bass page / arrangement. No on-screen text.
usage (agt folder): python3 -I var/seg_agt.py out.mp4"""
import sys, os, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); import agtdefs; G = agtdefs.G
V, cov, BEAT = G["V"], G["coverage"], G["BEAT"]
W, H, FPS = 1080, 1920, 30; NF = int(round(64 * BEAT * FPS)); GAP = 12; FACE_H = 690; GREY = (17, 17, 17)
DROP = 32 * BEAT
def tau(view, s0, s1):
    t = cov(view, s0, s1)
    if t is not None: return t
    for d in np.arange(0.05, 1.0, 0.05):                 # recording starts a moment late: hold its first frame
        t = cov(view, s0 + d, s1)
        if t is not None: return t - d
    raise SystemExit(f"no coverage {view} {s0:.2f}-{s1:.2f}")
class Run:
    def __init__(s, view, b0, b1, rect, track=False):
        s.view, s.t0, s.t1, s.rect, s.track = view, b0 * BEAT, b1 * BEAT, rect, track
        s.tau = tau(view, s.t0, s.t1); s.p = None; s.last = None; s.xs = None
    def frame(s, n):
        if s.p is None:
            a, b, _ = V[s.view]["segs"][0]; ss = s.tau
            s.hold = max(0, int(round((a - ss) * FPS))) if ss < a else 0
            s.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{max(ss, a):.3f}", "-i", V[s.view]["clip"], "-vf", f"fps={FPS}",
                                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        if s.hold > 0 and s.last is not None: s.hold -= 1; return s.last
        buf = s.p.stdout.read(1920 * 1200 * 3)
        if len(buf) == 1920 * 1200 * 3: s.last = np.frombuffer(buf, np.uint8).reshape(1200, 1920, 3)
        if s.hold > 0: s.hold -= 1
        return s.last
    def crop(s, n, size):
        f = s.frame(n); x, y, w, h = s.rect
        if s.track:                                     # page-mode bass in a narrower strip: follow FL's teal playhead
            g = f[200:900, 92:1893].astype(int); m = ((g[..., 0] > 90) & (g[..., 1] > 90) & (g[..., 2] < 70)).sum(0)
            if m.max() > 400:
                ph = 92 + int(np.argmax(m)); tgt = min(max(ph - 0.4 * w, 92), 1893 - w)
                s.xs = tgt if s.xs is None or abs(tgt - s.xs) > 500 else 0.9 * s.xs + 0.1 * tgt
            x = int(round(s.xs if s.xs is not None else x))
        return cv2.resize(f[y:y + h, x:x + w], size, interpolation=cv2.INTER_AREA)
BASS_R3 = (92, 250, 1801, 667); MAC_R3 = (772, 200, 1120, 413); CH_R3 = (740, 347, 1153, 425); SH_R3 = (375, 393, 1215, 450)
SH2 = (H - FACE_H - 2 * GAP) // 2                       # 603: two strips in bars 1-8
BASS_R2 = (92, 186, 1343, 750); MAC_R2 = (772, 190, 1120, 625)
sec1 = [[Run("BASS", 0, 16, BASS_R2, True), Run("BASS_4", 16, 32, BASS_R2, True)], [Run("MACRO2", 0, 32, MAC_R2)]]
sec2 = [[Run("SHAPER", 32, 37, SH_R3), Run("CHORDS2", 37, 64, CH_R3)],
        [Run("BASS_8", 32, 48, BASS_R3), Run("BASS_12", 48, 64, BASS_R3)], [Run("MACRO2", 32, 64, MAC_R3)]]
FACE_STARTS = [9.9, 13.33, 18.6, 22.9, 5.0, 14.25, 10.4, 18.65]      # listening stretches, one per 2 bars
fpipe = [None, -1]
def face(n):
    k = int(n / (8 * BEAT * FPS))
    if k != fpipe[1]:
        if fpipe[0]: fpipe[0].kill()
        fpipe[0] = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{FACE_STARTS[k]:.3f}", "-i", "face_pac.mp4", "-vf", f"fps={FPS},scale=1080:1920",
                                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE); fpipe[1] = k
    f = np.frombuffer(fpipe[0].stdout.read(1080 * 1920 * 3), np.uint8).reshape(1920, 1080, 3)
    return f[500:500 + FACE_H]
def run_at(runs, t): return next(r for r in runs if r.t0 - 1e-6 <= t < r.t1 + 1e-6)
out = sys.argv[1]
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-ss", "0", "-t", f"{NF / FPS:.3f}", "-i", "beat.mp3", "-map", "0:v", "-map", "1:a",
                        "-af", f"afade=t=out:st={NF / FPS - 0.12:.3f}:d=0.12", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-shortest", out], stdin=subprocess.PIPE)
nd = int(round(DROP * FPS)); n37 = int(round(37 * BEAT * FPS))
for n in range(NF):
    t = n / FPS; img = np.empty((H, W, 3), np.uint8); img[:] = GREY
    img[:FACE_H] = face(n); y = FACE_H + GAP
    slots = sec1 if t < DROP - 1e-6 else sec2; hh = SH2 if slots is sec1 else (H - FACE_H - 3 * GAP) // 3
    for i, runs in enumerate(slots):
        r = run_at(runs, t); h_ = hh if i < len(slots) - 1 else H - y
        strip = r.crop(n, (W, h_))
        if slots is sec2 and i == 0 and 0 <= n - n37 < 4:            # the ShaperBox -> chords swap: a quick flash on that strip
            a = (0.6, 0.35, 0.15, 0.05)[n - n37]; strip = cv2.addWeighted(strip, 1 - a, np.full_like(strip, 255), a, 0)
        img[y:y + h_] = strip; y += h_ + GAP
    if 0 <= n - nd < 6 or nd - n in (1, 2):                            # warm light flash on the drop
        a = {nd - 2: 0.25, nd - 1: 0.6}.get(n, (0.9, 0.7, 0.45, 0.25, 0.12, 0.05)[min(max(n - nd, 0), 5)])
        warm = np.empty_like(img); warm[:] = (200, 235, 255)
        img = cv2.addWeighted(img, 1 - a, warm, a, 0)
    enc.stdin.write(img.tobytes())
enc.stdin.close(); enc.wait(); print("frames", NF)
