"""Apply Jack's CapCut filter (measured as an affine BGR transform, cc/cc_filter_fit.json) to a rendered edit; audio copied.
usage: cc_filter.py in.mp4 out.mp4 fit.json"""
import sys, json, subprocess, numpy as np
SRC, OUT, FIT = sys.argv[1:4]; W, H = 1080, 1920
M = np.array(json.load(open(FIT))["M"], np.float32)          # 4x3: [B G R 1] -> BGR
lut_in = np.arange(256, dtype=np.float32)
rd = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", SRC,
                       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", "-c:a", "copy", "-shortest", OUT], stdin=subprocess.PIPE)
n = 0
while True:
    b = rd.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    f = np.frombuffer(b, np.uint8).reshape(-1, 3).astype(np.float32)
    o = f @ M[:3] + M[3]
    wr.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes()); n += 1
wr.stdin.close(); wr.wait(); print("frames", n)
