"""Pixel-exact note-edit finder: pink note pixels (R>200, R-G>120) in the grid, frame vs 6 frames earlier, ignoring
+-30 px around FL's playhead marker; page flips (marker jump > 60 px) skipped. usage: df_editpx.py clip..."""
import subprocess, sys, json, numpy as np
for clip in sys.argv[1:]:
    S = json.load(open(f"{clip}_scan.json"))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", clip + ".mp4", "-vf", "crop=1539:757:354:180,scale=770:379", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    M = []
    while True:
        b = p.stdout.read(770 * 379 * 3)
        if len(b) < 770 * 379 * 3: break
        f = np.frombuffer(b, np.uint8).reshape(379, 770, 3).astype(np.int16)
        M.append((f[..., 2] > 200) & (f[..., 2] - f[..., 1] > 120))
    ev = []
    for i in range(6, len(M)):
        p0, p1 = S[i - 6][0], S[i][0]
        if p0 is not None and p1 is not None and abs(p1 - p0) > 60: continue
        d = M[i] ^ M[i - 6]
        for pp in (p0, p1):
            if pp is not None:
                c = int((pp - 354) / 2); d[:, max(0, c - 15):c + 15] = False
        n = int(d.sum())
        if n > 25: ev.append((i / 60, n))
    m = []
    for t, n in ev:
        if m and t - m[-1][1] < 0.25: m[-1][1] = t; m[-1][2] = max(m[-1][2], n)
        else: m.append([t, t, n])
    print(clip, [(round(a, 2), round(b, 2), n) for a, b, n in m], flush=True)
