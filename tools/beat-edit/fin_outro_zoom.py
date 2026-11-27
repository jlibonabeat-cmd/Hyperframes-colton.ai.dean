"""Outro on the last shot (Jack, Nov 27: "a dynamic zoom into the midi notes with a bit of motion blur following it and
then an exit transition"). On the bar downbeat the whole frame punches into the notes at the playhead (face panel pushes
out the top), keeps riding the playhead with a slow push, then rushes out with a zoom-blur + chromatic split + warm-white
flash on the last frames, which loops into the opening impact zoom-out. Audio copied.
usage: fin_outro_zoom.py in.mp4 out.mp4"""
import sys, subprocess, numpy as np, cv2

SRC, OUT = sys.argv[1:3]
W, H = 1080, 1920
N_SHOT = 597              # last shot (face split over the bass roll) starts here
N_PUNCH = 606             # bar-32 downbeat (beat 64): the punch-in
PUNCH_LEN = 9
N_EXIT = 630              # exit rush starts
Z_PUNCH, Z_HOLD, Z_EXIT = 2.15, 2.4, 5.0
NOTE_Y = 1250             # between the G#4 (y~1030) and C4 (y~1456) rows
TARGET_Y = 880            # where that point lands after the zoom (~46% of the frame, clear of the IG caption zone)


def ease_out(u):
    u = min(max(u, 0), 1); return 1 - (1 - u) ** 3


def ease_in(u):
    u = min(max(u, 0), 1); return u ** 2.2


def params(t, nf):
    """t = fractional frame index -> (z, cx, cy, px, py, exit 0..1)."""
    ph = 587 + (t - 597) * 7.3                                   # measured playhead x (it travels right ~7.3 px/frame)
    if t < N_PUNCH:
        z = 1.0
    elif t < N_PUNCH + PUNCH_LEN:
        z = 1 + (Z_PUNCH - 1) * ease_out((t - N_PUNCH) / PUNCH_LEN)
    elif t < N_EXIT:
        z = Z_PUNCH + (Z_HOLD - Z_PUNCH) * (t - N_PUNCH - PUNCH_LEN) / (N_EXIT - N_PUNCH - PUNCH_LEN)
    else:
        z = Z_HOLD + (Z_EXIT - Z_HOLD) * ease_in((t - N_EXIT) / (nf - 1 - N_EXIT))
    a = min(1.0, (z - 1) / (Z_PUNCH - 1))
    half = 540 / z
    cx = min(max(ph - 60, half), W - half)                        # keep the playhead a touch right of centre, crop inside
    cx = 540 + (cx - 540) * a
    cy = NOTE_Y
    px = 540 + (cx - 540) * (1 - a)
    py = cy + (TARGET_Y - cy) * a
    ex = 0.0 if t < N_EXIT else (t - N_EXIT) / (nf - 1 - N_EXIT)
    return z, cx, cy, px, py, ex


def warp(img, p):
    z, cx, cy, px, py, _ = p
    M = np.float32([[z, 0, px - z * cx], [0, z, py - z * cy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
nf = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                  "stream=nb_read_frames", "-of", "csv=p=0", SRC]).decode().strip())
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", SRC,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", "-c:a", "copy", OUT], stdin=subprocess.PIPE)
n = 0
while True:
    b = rd.stdout.read(W * H * 3)
    if len(b) < W * H * 3:
        break
    img = np.frombuffer(b, np.uint8).reshape(H, W, 3)
    if n >= N_PUNCH:
        p0, p1 = params(n - 1, nf), params(n, nf)
        # motion blur: average the path from the last frame to this one, enough samples that it streaks (no stepping):
        # roughly one sample per 6 px of movement at the frame edge, 3..48
        move = max(abs(p1[0] - p0[0]) * 1100 + abs(p1[3] - p0[3]) + abs(p1[4] - p0[4]) + abs(p1[1] - p0[1]) * p1[0], 0)
        k = int(min(48, max(3, move / 6)))
        acc = np.zeros((H, W, 3), np.float32)
        for u in np.linspace(0, 1, k):
            q = tuple(a_ + (b_ - a_) * u for a_, b_ in zip(p0, p1))
            acc += warp(img, q)
        f = acc / k / 255
        ex = p1[5]
        if ex > 0:                                              # exit: chromatic split + warm-white flash building to the end
            px_ = int(round(14 * ex))
            if px_:
                f[..., 2] = np.roll(f[..., 2], px_, 1); f[..., 0] = np.roll(f[..., 0], -px_, 1)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            glow = np.exp(-(((xx - 540) / 520) ** 2 + ((yy - TARGET_Y) / 760) ** 2))[..., None]
            warmc = np.array([0.82, 0.93, 1.0], np.float32)
            a = (ex ** 2.0) * 0.9 * (0.55 + 0.45 * glow)
            f = f * (1 - a) + warmc * a
        img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    wr.stdin.write(np.ascontiguousarray(img).tobytes()); n += 1
wr.stdin.close(); wr.wait(); print("frames", n, "of", nf)
