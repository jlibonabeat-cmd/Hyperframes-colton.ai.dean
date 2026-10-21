"""Detect the FL playhead (thin full-height green vertical line) per frame in a screen recording."""
import json, subprocess, sys
import numpy as np
clip = sys.argv[1]
FPSD = 20
W, H = 960, 540   # half-res
raw = subprocess.run(["ffmpeg", "-nostats", "-v", "error", "-i", clip, "-vf", f"fps={FPSD},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                     capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.int16)
xs = []
for f in fr:
    reg = f[110:440]                       # roll/playlist body (half-res rows)
    g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(axis=0)
    cand = np.where(col > 0.75)[0]
    cand = cand[cand > 200]                # skip browser panel / piano keys
    xs.append(int(cand[0]) * 2 if len(cand) else -1)
json.dump(xs, open(sys.argv[2], "w"))
xs = np.array(xs)
vis = xs >= 0
print(clip.split("/")[-1], "frames", len(xs), "visible", int(vis.sum()))
# print compact timeline: time -> x
last = None
for i, x in enumerate(xs):
    if i % 10 == 0:
        print(f"{i / FPSD:5.1f}:{x}", end="  ")
print()
