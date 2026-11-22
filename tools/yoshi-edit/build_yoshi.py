"""YOSHI EDIT ("clone / disappearing", after @prodyoshi01) - Something2say (144 BPM, F# minor), 0:00-0:17.30.
One locked-off phone shot: plate.mp4 = the empty room (IMG_9930), act.mp4 = Jack in it (IMG_9931) warped onto the plate
(homography, 2.6 % zoom difference) and wall-matched. He only stands in one spot, so this is the disappearing version:
hard cuts on the beat between the empty room and different moments of him (no clones - they'd overlap). No camera
moves, no grade, no text. Drop = the kick + snare hit at 3.97 s; a snare every bar (1.667 s) after it.
usage: VARIANT=A|B python3 build_yoshi.py out.mp4"""
import cv2, numpy as np, subprocess, sys, os
FPS = 30; END = 17.30; W, H = 1080, 1920
VAR = os.environ.get("VARIANT", "A")
# poses (act.mp4 time where the moment starts): A playing, looking at the screen (blue desk light); B head up to the lens;
# C looking back over his right shoulder; D hand on keys looking right; E turned to the guitars; F playing, warm light
P = dict(A=0.4, A2=2.4, B=4.6, C=6.5, D=9.2, D2=11.0, E=14.0, E2=17.0, F=22.0, F2=24.5)
E_ = None
if VAR == "A":
    # (start, source, pose) ; source None = empty room
    E_ = [(0.00, None, 0), (3.14, "A", 0), (3.24, None, 0), (3.35, "B", 0), (3.45, None, 0), (3.55, "C", 0), (3.66, None, 0),
          (3.76, "E", 0), (3.86, None, 0),
          (3.97, "A", 0), (5.64, None, 0), (5.94, "C", 0), (7.30, "D", 0), (8.97, None, 0), (9.28, "E", 0),
          (10.64, "F", 0), (12.30, None, 0), (12.61, "A2", 0),
          (13.28, None, 0), (13.33, "B", 0), (13.41, None, 0), (13.50, "D2", 0), (13.58, None, 0), (13.66, "E2", 0), (13.75, None, 0),
          (13.97, "F2", 0), (15.64, None, 0), (15.94, "D", 0), (16.78, None, 0)]
else:
    # B: longer holds, two-bar intro empty then he appears on the drop with no pre-roll pops; poses swap on the snares
    E_ = [(0.00, None, 0), (3.97, "F", 0), (5.73, "C", 0), (7.40, None, 0), (7.73, "E", 0), (9.06, "B", 0),
          (10.73, None, 0), (10.94, "D", 0), (12.39, "A", 0),
          (13.28, None, 0), (13.41, "E2", 0), (13.50, None, 0), (13.58, "F2", 0),
          (15.64, None, 0), (15.94, "D2", 0), (16.78, None, 0)]
caps = {"plate": cv2.VideoCapture("plate.mp4"), "act": cv2.VideoCapture("act.mp4")}
def read(name, t):
    c = caps[name]; n = int(round(t * FPS))
    if int(c.get(cv2.CAP_PROP_POS_FRAMES)) != n: c.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = c.read(); return f
out = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-ss", "0", "-t", str(END), "-i", "beat.mp3", "-map", "0:v", "-map", "1:a",
                        "-af", f"afade=t=out:st={END - 0.12}:d=0.12", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", sys.argv[1]], stdin=subprocess.PIPE)
NF = int(round(END * FPS)); log = []
for n in range(NF):
    t = n / FPS
    k = max(i for i, e in enumerate(E_) if e[0] <= t + 0.5 / FPS); st, src, _ = E_[k]   # nearest frame
    if src is None:
        f = read("plate", t)                       # the empty room plays continuously in song time
    else:
        f = read("act", P[src] + (t - st))       # each appearance plays on from its pose moment
    out.stdin.write(f.tobytes())
    if not log or log[-1][1] != (src or "EMPTY"): log.append((round(t, 2), src or "EMPTY"))
out.stdin.close(); out.wait()
print(VAR, len(log), "states:", log)
