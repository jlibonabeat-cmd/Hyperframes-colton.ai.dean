"""Note-edit finder for smooth-follow rolls: note mask (pale green) vs 6 frames earlier, best horizontal shift
compensated; residual > thr = something changed that isn't scrolling. usage: s2s_editscroll.py clip..."""
import subprocess, sys, json, numpy as np
for clip in sys.argv[1:]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-vf", "crop=1372:756:521:183,scale=686:378", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    M = []
    while True:
        b = p.stdout.read(686 * 378 * 3)
        if len(b) < 686 * 378 * 3: break
        f = np.frombuffer(b, np.uint8).reshape(378, 686, 3).astype(np.int16)
        M.append((f[..., 1] > 180) & (f[..., 2] > 120) & (f[..., 1] - f[..., 2] > 15) & (f[..., 1] - f[..., 0] > 5))
    C = json.load(open("clock2.json"))[clip]["runs"]
    ev = []
    for i in range(6, len(M), 3):
        a, b = M[i - 6], M[i]; best = None
        for sh in range(0, 60):
            r = (np.roll(a, -sh, axis=1)[:, 5:620] ^ b[:, 5:620]).sum()
            if best is None or r < best[0]: best = (r, sh)
        if best[0] > 80: ev.append((round(i / 60, 2), int(best[0]), best[1]))
    m = []
    for t, r, sh in ev:
        if m and t - m[-1][1] < 0.3: m[-1][1] = t; m[-1][2] = max(m[-1][2], r)
        else: m.append([t, t, r])
    print(clip, "runs", [(round(a, 2), round(b, 2)) for a, b, *_ in C], "\n  changes:", [(a, b, r) for a, b, r in m], flush=True)
