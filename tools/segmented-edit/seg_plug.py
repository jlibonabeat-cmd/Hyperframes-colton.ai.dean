"""REALLY I'M THE PLUG finishing pass.
A: impact start (fast motion-blurred zoom out, frames 0-13) on the J Sewell render.
B: bars 1-4 (0-7.68 s) = SEGMENTED section - graded face on top, three stacked in-sync strips: the bell's piano-roll page
   with FL's playhead sweeping it / the wet-sax track scrolling through the parked playhead / the Confession hits
   (bars 1-2) swapping (flash) to the zoomed arrangement at bar 3; slow push on every strip. Impact start, then a zoom-blur
   punch out of the strips into the J Sewell render from bar 5 (whose incoming zoom + punch-in already land on 7.68).
usage (plug folder): python3 -I scripts/seg_plug.py A|B in_js.mp4 out.mp4"""
import sys, json, subprocess, numpy as np, cv2
MODE, SRC, OUT = sys.argv[1:4]
W, H, FPS = 1080, 1920, 30; BEAT = 60 / 125; NF = int(round(32 * BEAT * FPS)); ND = int(round(16 * BEAT * FPS))   # 461, 230
FACE_H, GAP = 690, 12; GREY = (22, 22, 22)
def ease(u): u = min(max(u, 0), 1); return u * u * (3 - 2 * u)
def scale(img, z):
    hh, ww = img.shape[:2]; M = cv2.getRotationMatrix2D((ww / 2, hh / 2), 0, z)
    return cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def zblur(img, z0, z1, k=10):
    acc = np.zeros(img.shape, np.float32)
    for z in np.linspace(z0, z1, k): acc += scale(img, z)
    return (acc / k).astype(np.uint8)
def push(r, z): x, y, w, h = r; return (x + w * (1 - 1 / z) / 2, y + h * (1 - 1 / z) / 2, w / z, h / z)
def crop(f, r, size): x, y, w, h = [int(round(v)) for v in r]; return cv2.resize(f[y:y + h, x:x + w], size, interpolation=cv2.INTER_AREA)
class Src:
    def __init__(s, path, D, a, b): s.path, s.D, s.a, s.b, s.p, s.n, s.last = path, D, a, b, None, -1, None
    def at(s, t):
        c = min(max(t - s.D, s.a), s.b); k = int(round(c * FPS))
        if s.p is None:
            s.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{k / FPS:.4f}", "-i", s.path, "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE); s.n = k - 1
        while s.n < k:
            b = s.p.stdout.read(1920 * 1200 * 3)
            if len(b) < 1920 * 1200 * 3: break
            s.last = np.frombuffer(b, np.uint8).reshape(1200, 1920, 3); s.n += 1
        return s.last
def frames(path, w, h, ss=0.0, vf=None):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-i", path] + (["-vf", vf] if vf else []) + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3: return
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
js = frames(SRC, W, H)
if MODE == "B":
    BELL = Src("bell.mp4", -7.715, 7.75, 15.45); SAX = Src("sax.mp4", -0.87, 0.8, 31.45)
    BASS = Src("bass.mp4", -1.03, 1.05, 12.7); MZ = Src("macroz.mp4", -0.965, 0.9, 31.65)
    face = frames("face_graded.mp4", 1080, 1920, 14.8, f"fps={FPS}")
    SH = (H - FACE_H - 3 * GAP) // 3                       # 406 per strip, the last takes the remainder
    AR = W / SH
    def R(x0, x1, cy): w = x1 - x0; h = w / AR; return (x0, cy - h / 2, w, h)
    R_BELL = R(470, 1800, 575)                                 # the whole bell page (bars 1-5), the playhead sweeps it
    R_SAX = R(560, 1890, 815)                                  # wet sax (track 9) through the parked playhead (x 886)
    R_BASS = R(560, 1890, 545)                                 # Confession hits (tracks 12/13)
    R_MZ = R(560, 1890, 660)                                   # zoomed arrangement
    n8 = int(round(8 * BEAT * FPS))
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-t", f"{NF / FPS:.3f}", "-i", "beat.mp3", "-map", "0:v", "-map", "1:a", "-af", f"afade=t=out:st={NF / FPS - 0.12:.3f}:d=0.12",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", OUT], stdin=subprocess.PIPE)
for n in range(NF):
    jf = next(js); img = jf.copy()
    if MODE == "B" and n < ND:
        t = n / FPS; img = np.empty((H, W, 3), np.uint8); img[:] = GREY
        fr = next(face); img[:FACE_H] = fr[380:380 + FACE_H]; y = FACE_H + GAP
        z = 1 + 0.07 * ease(t / (ND / FPS))                    # J Sewell: slow push through the section
        third = (MZ, R_MZ) if n >= n8 else (BASS, R_BASS)
        for i, (src, r) in enumerate(((BELL, R_BELL), (SAX, R_SAX), third)):
            h = SH if i < 2 else H - y
            strip = crop(src.at(t), push(r, z), (W, h))
            if i == 2 and 0 <= n - n8 < 5:                     # the strip swap at bar 3: quick white flash
                a = (0.7, 0.45, 0.25, 0.12, 0.05)[n - n8]; strip = cv2.addWeighted(strip, 1 - a, np.full_like(strip, 255), a, 0)
            img[y:y + h] = strip; y += h + GAP
        if ND - 3 <= n:                                        # punch out of the strips into bar 5 (zoom-blur, J Sewell)
            img = zblur(img, 1.0, (1.1, 1.22, 1.35)[n - ND + 3], 9)
    if n < 14:                                                 # impact start: opens zoomed in, snaps out with motion blur
        z0 = 1 + 0.42 * (1 - ease(n / 14)) ** 1.6; z1 = 1 + 0.42 * (1 - ease((n + 1) / 14)) ** 1.6
        img = zblur(img, z1, z0, 10) if n < 9 else scale(img, z0)
    enc.stdin.write(np.ascontiguousarray(img).tobytes())
enc.stdin.close(); enc.wait(); print("frames", NF)
