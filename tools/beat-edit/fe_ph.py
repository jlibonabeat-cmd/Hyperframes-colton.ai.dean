"""FEAR: add the playhead x to {clip}_scan.json. The playhead is a thin dark-teal line (~R9 G43 B46) over the grid."""
import subprocess, json, sys, numpy as np
W, H = 1920, 1200
for clip in sys.argv[1:]:
    S = json.load(open(clip + "_scan.json"))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-vf", "crop=1320:740:568:195", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    out = []; i = 0
    while True:
        buf = p.stdout.read(1320 * 740 * 3)
        if len(buf) < 1320 * 740 * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(740, 1320, 3)[::3].astype(np.int16)
        B, G, R = f[..., 0], f[..., 1], f[..., 2]
        m = (R < 28) & (G > 36) & (B > 36) & (G < 90) & (B < 90) & (np.abs(B - G) < 20)
        cov = m.mean(0); xs = np.where(cov > 0.5)[0]
        x = float(np.median(xs)) + 568 if len(xs) and xs.max() - xs.min() < 12 else None
        if i < len(S): S[i] = (x, S[i][1])
        out.append(x); i += 1
    json.dump(S, open(clip + "_scan.json", "w"))
    v = [x for x in out if x is not None]
    print(clip, "playhead found in", len(v), "/", len(out), "frames; x range", (min(v), max(v)) if v else None,
          "samples", [out[k] for k in range(0, len(out), max(1, len(out) // 10))])
