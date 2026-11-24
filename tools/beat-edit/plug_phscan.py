"""Per-frame FL playhead x (green column, blue-grey theme) for playlist / roll recordings -> {clip}_ph.json (60 fps list).
usage: plug_phscan.py clip..."""
import subprocess, sys, json, numpy as np
for clip in sys.argv[1:]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{clip}.mp4", "-vf", "crop=1450:600:450:420", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(1450 * 600 * 3)
        if len(b) < 1450 * 600 * 3: break
        g = np.frombuffer(b, np.uint8).reshape(600, 1450, 3)[::3].astype(np.int16)
        R, G, B = g[..., 2], g[..., 1], g[..., 0]
        m = ((G > 140) & (G - R > 35) & (G - B > 50)).mean(0)
        xs = np.where(m > 0.45)[0]
        out.append(float(np.median(xs)) + 450 if len(xs) and xs.max() - xs.min() < 10 else None)
    json.dump(out, open(f"{clip}_ph.json", "w"))
    v = [x for x in out if x is not None]
    print(clip, len(out), "found", len(v), "x range", (min(v), max(v)) if v else None)
