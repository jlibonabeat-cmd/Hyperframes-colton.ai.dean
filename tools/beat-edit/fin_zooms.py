"""Dynamic note zooms on a rendered edit (Jack, Nov 27: "a dynamic zoom into the midi notes with a bit of motion blur
following it and then an exit transition", then "that same zoom" on the bass shots). Per zoom: on a downbeat the whole
frame punches into the notes at the playhead (eased, motion blur along the path; on a split the face panel pushes out the
top), then a slow push riding the playhead; optionally an exit rush (zoom to 5x, heavy zoom blur, chromatic split,
warm-white flash). The playhead path is measured from the input frames. Audio copied.
usage: fin_zooms.py in.mp4 out.mp4"""
import sys, subprocess, numpy as np, cv2

SRC, OUT = sys.argv[1:3]
W, H = 1080, 1920
TARGET_Y = 880            # where the note point lands (~46% of the frame, clear of the IG caption zone)
# punch = first frame of the punch-in (a downbeat), end = first frame after the shot, exit = first frame of the exit rush
# note_y = the note rows' centre (frame y), ph_off = zoom centre relative to the playhead (+ = show the notes coming)
ZOOMS = [
    # Mystery v6, 0:01-0:05 bass going in (full): bar-19 downbeat, lands as the hats come in; C4 row y 1430-1495
    dict(punch=114, plen=9, end=156, exit=None, z1=2.0, z2=2.2, note_y=1462, ph_off=90),
    # section-2 bass entry (split), the end-of-8-bar notes going in (G#4/G4 y ~1000-1095, C4 y ~1450); exits on bar 9
    dict(punch=315, plen=9, end=351, exit=341, z1=2.15, z2=2.3, note_y=1235, ph_off=0),
    # last shot (split): bar-32 downbeat, exit rush loops into the opening zoom-out
    dict(punch=606, plen=9, end=None, exit=630, z1=2.15, z2=2.4, note_y=1250, ph_off=-60),
]
Z_EXIT = 5.0


def ease_out(u):
    u = min(max(u, 0), 1); return 1 - (1 - u) ** 3


def ease_in(u):
    u = min(max(u, 0), 1); return u ** 2.2


def playhead_x(f):
    d = f[850:1800].astype(np.int16)
    m = (d[..., 1] > 150) & (d[..., 0] < 90) & (d[..., 2] < 170)
    col = m.sum(0)
    return float(np.argmax(col)) if col.max() > 300 else None


rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
frames = []
while True:
    b = rd.stdout.read(W * H * 3)
    if len(b) < W * H * 3:
        break
    frames.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
nf = len(frames)
for zc in ZOOMS:
    zc["end"] = zc["end"] or nf
    ns = np.arange(zc["punch"] - 1, zc["end"])
    xs = [playhead_x(frames[n]) for n in ns]
    ok = [(n, x) for n, x in zip(ns, xs) if x is not None]
    a_, b_ = np.polyfit([o[0] for o in ok], [o[1] for o in ok], 1)   # the playhead travels linearly (or sits still)
    zc["ph"] = (float(a_), float(b_))
    print(f"zoom @{zc['punch']}: playhead x = {a_:.2f} n + {b_:.0f} ({len(ok)} frames)")


def params(zc, t):
    """t = fractional frame -> (z, cx, cy, px, py, exit 0..1)"""
    p0, pl, ex0, end = zc["punch"], zc["plen"], zc["exit"], zc["end"]
    hold_end = ex0 if ex0 else end
    if t < p0:
        z = 1.0
    elif t < p0 + pl:
        z = 1 + (zc["z1"] - 1) * ease_out((t - p0) / pl)
    elif t < hold_end:
        z = zc["z1"] + (zc["z2"] - zc["z1"]) * (t - p0 - pl) / max(1, hold_end - p0 - pl)
    else:
        z = zc["z2"] + (Z_EXIT - zc["z2"]) * ease_in((t - ex0) / max(1, end - 1 - ex0))
    a = min(1.0, (z - 1) / (zc["z1"] - 1))
    ph = zc["ph"][0] * t + zc["ph"][1] + zc["ph_off"]
    half = 540 / z
    cx = 540 + (min(max(ph, half), W - half) - 540) * a
    cy = zc["note_y"]
    px = 540 + (cx - 540) * (1 - a)
    py = cy + (TARGET_Y - cy) * a
    ex = 0.0 if not ex0 or t < ex0 else (t - ex0) / max(1, end - 1 - ex0)
    return z, cx, cy, px, py, ex


def warp(img, p):
    z, cx, cy, px, py, _ = p
    tx = min(0.0, max(W - z * W, px - z * cx))                  # keep the frame covered (no border)
    ty = min(0.0, max(H - z * H, py - z * cy))
    M = np.float32([[z, 0, tx], [0, z, ty]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
GLOW = np.exp(-(((xx - 540) / 520) ** 2 + ((yy - TARGET_Y) / 760) ** 2))[..., None]
WARM = np.array([0.82, 0.93, 1.0], np.float32)

wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", SRC,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", "-c:a", "copy", OUT], stdin=subprocess.PIPE)
for n, img in enumerate(frames):
    zc = next((z_ for z_ in ZOOMS if z_["punch"] <= n < z_["end"]), None)
    if zc is not None:
        p0, p1 = params(zc, n - 1), params(zc, n)
        # motion blur along the path from the last frame to this one: ~1 sample per 6 px of edge movement, 3..48
        move = abs(p1[0] - p0[0]) * 1100 + abs(p1[3] - p0[3]) + abs(p1[4] - p0[4]) + abs(p1[1] - p0[1]) * p1[0]
        k = int(min(48, max(3, move / 6)))
        acc = np.zeros((H, W, 3), np.float32)
        for u in np.linspace(0, 1, k):
            acc += warp(img, tuple(a_ + (b_ - a_) * u for a_, b_ in zip(p0, p1)))
        f = acc / k / 255
        ex = p1[5]
        if ex > 0:                                              # exit: chromatic split + warm-white flash
            px_ = int(round(14 * ex))
            if px_:
                f[..., 2] = np.roll(f[..., 2], px_, 1); f[..., 0] = np.roll(f[..., 0], -px_, 1)
            a = (ex ** 2.0) * 0.9 * (0.55 + 0.45 * GLOW)
            f = f * (1 - a) + WARM * a
        img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    wr.stdin.write(np.ascontiguousarray(img).tobytes())
wr.stdin.close(); wr.wait(); print("frames", nf)
