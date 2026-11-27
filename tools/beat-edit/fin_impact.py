"""Impact start (fast motion-blurred zoom out over the first 14 frames) on a rendered edit; audio copied.
usage: fin_impact.py in.mp4 out.mp4"""
import sys, subprocess, numpy as np, cv2
SRC, OUT = sys.argv[1:3]; W, H = 1080, 1920
def ease(u): u = min(max(u, 0), 1); return u * u * (3 - 2 * u)
def scale(img, z):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z); return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def zblur(img, z0, z1, k=10):
    acc = np.zeros(img.shape, np.float32)
    for z in np.linspace(z0, z1, k): acc += scale(img, z)
    return (acc / k).astype(np.uint8)
rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", SRC,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                       "-c:a", "copy", "-shortest", OUT], stdin=subprocess.PIPE)
n = 0
while True:
    b = rd.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    img = np.frombuffer(b, np.uint8).reshape(H, W, 3)
    if n < 14:
        z0 = 1 + 0.42 * (1 - ease(n / 14)) ** 1.6; z1 = 1 + 0.42 * (1 - ease((n + 1) / 14)) ** 1.6
        img = zblur(img, z1, z0) if n < 9 else scale(img, z0)
    wr.stdin.write(np.ascontiguousarray(img).tobytes()); n += 1
wr.stdin.close(); wr.wait(); print("frames", n)
