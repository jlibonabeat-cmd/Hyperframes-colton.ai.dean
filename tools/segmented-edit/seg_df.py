"""DRUG FIT - SEGMENTED edit (Jack, Nov 24): face on top, stacked in-sync piano-roll strips below, no text.
Bars 1-8 (0:00-0:10.61): 3 strips - snare pattern (whole page) / open hats going in on the playhead / Spinz 808 going in.
Drop (bar 9): warm flash; the mixer takes the strip area (all meters slamming), the 808 main roll slides in as a
follow-along strip riding FL's playhead in its clean stretches (beats 37.5-43.8, 56.3-64; Jack redraws its last notes
in between, kept off screen). Sync: FL's clock OCR (df_clock.py), song time = mp3 time (B0 = 0).
usage (df folder): python3 -I seg_df.py out.mp4"""
import sys, subprocess, json, numpy as np, cv2
W, H, FPS = 1080, 1920, 30
BEAT = 60 / 181; NF = int(round(64 * BEAT * FPS)); DROP = 32 * BEAT
FACE_H, GAP = 690, 12; GREY = (24, 24, 24)
class Src:
    """sequential reader: frame for song time t (clip = t - D), clamped to the clean run [a, b] (holds the edge frame)"""
    def __init__(s, path, D, a, b):
        s.path, s.D, s.a, s.b = path, D, a, b; s.p = None; s.n = -1; s.last = None
    def at(s, t):
        c = min(max(t - s.D, s.a), s.b); k = int(round(c * FPS))
        if s.p is None:
            s.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{k / FPS:.4f}", "-i", s.path, "-vf", f"fps={FPS}", "-f", "rawvideo",
                                    "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE); s.n = k - 1
        while s.n < k:
            buf = s.p.stdout.read(1920 * 1200 * 3)
            if len(buf) < 1920 * 1200 * 3: break
            s.last = np.frombuffer(buf, np.uint8).reshape(1200, 1920, 3); s.n += 1
        return s.last
SN = Src("snare.mp4", -3.29, 3.3, 13.85); OH = Src("ohat.mp4", -3.562, 3.55, 13.9); SP = Src("spinz.mp4", -4.936, 4.9, 15.45)
M8 = Src("m808.mp4", 4.195, 6.4, 16.9); MX = Src("mixer.mp4", -0.878, 0.9, 22.55)
# FL's playhead in the 808 roll: linear fit of the scanned marker (clip 6.4-15) -> smooth follow camera
sc = json.load(open("m808_scan.json")); pts = [(i / 60, r[0]) for i, r in enumerate(sc) if r and r[0] and 6.4 < i / 60 < 15]
PA, PB = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
def crop(f, r, size):
    x, y, w, h = [int(round(v)) for v in r]; return cv2.resize(f[y:y + h, x:x + w], size, interpolation=cv2.INTER_AREA)
def push(r, z):                                            # slow push: shrink the crop about its centre
    x, y, w, h = r; return (x + w * (1 - 1 / z) / 2, y + h * (1 - 1 / z) / 2, w / z, h / z)
def scale(img, z):
    hh, ww = img.shape[:2]; M = cv2.getRotationMatrix2D((ww / 2, hh / 2), 0, z)
    return cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def zblur(img, z0, z1, k=8):
    acc = np.zeros(img.shape, np.float32)
    for z in np.linspace(z0, z1, k): acc += scale(img, z)
    return (acc / k).astype(np.uint8)
def lerp(a, b, u): return [p + (q - p) * u for p, q in zip(a, b)]
def ease(u): u = min(max(u, 0), 1); return u * u * (3 - 2 * u)
S1 = (H - FACE_H - 2 * GAP) // 2                          # 603: v2 (Jack): 2 layers, the open-hat strip read as a duplicate 808
R_SN = (354, 278, 1539, 567); R_HT = (354, 286, 986, 986 * 603 / 1080)    # whole pages (hats / 808 patterns sit in the left 2/3)
M8H = 640                                                  # drop: 808 strip height when it's in
R_MX_FULL = (362, 150, 560, 1206 * 650 / 1080)             # mixer filling the strip area (master + inserts 1-10)
R_MX_HALF = (362, 175, 560, 566 * 560 / 1080)              # mixer as the lower strip (meters + names)
IN = [(37.5, 64.5)]          # v3 (Jack): once the 808 roll comes in (0:12.4) it stays; framed behind the playhead so his redraw of the last notes is off-crop until they go back in on the playhead
def m808_share(b):                                          # 0..1: how much of the strip area the 808 roll takes
    v = 0.0
    for a, e in IN:
        v = max(v, min(ease((b - a) / 0.75), ease((e - b) / 0.75)))
    return v
# face: listening stretches of the usual cap take (face_pac), cut on phrase lines
FACE = [(0, 12, 5.0), (12, 32, 9.9), (32, 40, 18.6), (40, 48, 22.9), (48, 56, 11.2), (56, 64.5, 5.4)]
fpipe = [None, None]
def face(n):
    b = n / FPS / BEAT; seg = next(x for x in FACE if x[0] <= b < x[1])
    if fpipe[1] is not seg:
        if fpipe[0]: fpipe[0].kill()
        fpipe[0] = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{seg[2]:.3f}", "-i", "face_pac.mp4", "-vf", f"fps={FPS}",
                                     "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE); fpipe[1] = seg
    return np.frombuffer(fpipe[0].stdout.read(1080 * 1920 * 3), np.uint8).reshape(1920, 1080, 3)[500:500 + FACE_H]
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-t", f"{NF / FPS:.3f}", "-i", "beat.mp3", "-map", "0:v", "-map", "1:a", "-af", f"afade=t=out:st={NF / FPS - 0.12:.3f}:d=0.12",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", sys.argv[1]], stdin=subprocess.PIPE)
nd = int(round(DROP * FPS)); xs = [None]; log = []; snx = [None]
_s = json.load(open("snare_scan.json")); SNS = [r[0] if r and r[0] else None for r in _s]
for _i in range(1, len(SNS)):
    if SNS[_i] is None: SNS[_i] = SNS[_i - 1]
for n in range(NF):
    t = n / FPS; b = t / BEAT
    img = np.empty((H, W, 3), np.uint8); img[:] = GREY; img[:FACE_H] = face(n); y0 = FACE_H + GAP
    if n < nd:                                               # bars 1-8: three strips
        for i, (src, r) in enumerate(((SN, R_SN), (SP, R_HT))):
            h = S1 if i < 1 else H - y0
            if src is SN:                                    # snare: follow-along crop riding FL's playhead (bigger hits)
                k = int(round((t - SN.D) * 60)); m = SNS[min(max(k, 0), len(SNS) - 1)]
                if m is not None: snx[0] = m if snx[0] is None or abs(m - snx[0]) > 250 else 0.75 * snx[0] + 0.25 * m
                w = 760; x = min(max((snx[0] or 400) - 0.3 * w, 354), 1893 - w); r = (x, 561 - w * h / W / 2, w, w * h / W)
            r = push(r, 1 + 0.07 * ease(t / DROP))             # J Sewell: slow push through the section
            img[y0:y0 + h] = crop(src.at(t), r, (W, h)); y0 += h + GAP
        if b >= 31:                                          # last beat before the drop: strobe (Jack's CapCut accent)
            if (n - int(31 * BEAT * FPS)) % 4 in (2, 3): img[FACE_H:] = (img[FACE_H:] * 0.35).astype(np.uint8)
    else:                                                    # the drop: 808 roll strip (when clean) over the mixer
        u = m808_share(b); area = H - y0; h8 = int(round(M8H * u)); hm = area - h8 - (GAP if h8 > 0 else 0)
        if h8 > 4:
            f8 = M8.at(t); clip = t - M8.D; ph = PA * clip + PB
            w8 = 420 * W / M8H; x8 = min(max(ph - 0.6 * w8, 354), 1893 - w8)
            xs[0] = x8 if xs[0] is None or abs(x8 - xs[0]) > 300 else 0.8 * xs[0] + 0.2 * x8
            full = crop(f8, push((xs[0], 346, w8, 420), 1 + 0.06 * ease((b - 37.5) / 26)), (W, M8H))
            img[y0:y0 + h8] = full[(M8H - h8) // 2:(M8H - h8) // 2 + h8]     # slides open from the middle
            y0 += h8 + GAP
        r = lerp(R_MX_FULL, R_MX_HALF, u); r[3] = hm * r[2] / W
        r[1] = min(max(r[1], 0), 1200 - r[3])
        img[y0:y0 + hm] = crop(MX.at(t), push(r, 1 + 0.06 * ease((b - 32) / 32)), (W, hm))
        log.append((round(t, 2), round(u, 2)))
    if n < 45:                                               # v3: slow zoom out on the intro (1.5 s)
        img = scale(img, 1 + 0.14 * (1 - ease(n / 45)))
    if nd - 3 <= n < nd:                                     # J Sewell transition into the drop: zoom-blur punch out...
        k = n - nd + 3; img = zblur(img, 1.0, (1.08, 1.18, 1.32)[k])
    elif nd <= n < nd + 8:                                   # ...and land: zoom-blurred punch-in that settles
        z = 1 + 0.22 * (1 - ease((n - nd) / 8)); img = zblur(scale(img, z), 1.0, 1 + 0.25 * (1 - (n - nd) / 8)) if n < nd + 4 else scale(img, z)
    if nd - 2 <= n < nd + 7:                                 # warm light flash on the drop (lighter under the punch)
        a = {nd - 2: 0.2, nd - 1: 0.45}.get(n, (0.7, 0.5, 0.32, 0.2, 0.1, 0.05, 0.02)[min(n - nd, 6)])
        warm = np.empty_like(img); warm[:] = (200, 235, 255); img = cv2.addWeighted(img, 1 - a, warm, a, 0)
    enc.stdin.write(img.tobytes())
enc.stdin.close(); enc.wait(); print("frames", NF, "playhead fit", round(PA, 1), "px/s")
