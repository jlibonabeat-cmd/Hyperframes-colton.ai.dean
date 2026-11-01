"""Ground-truth sync check on the RENDERED reel: in every piano-roll shot, find the frames where a note's left edge
crosses the (rendered) playhead, and compare those times to master onsets. Beige FL theme.
usage: wtw_rendercheck.py out/wtw_A.mp4 out/wtw_A_events.json beat.mp3 START"""
import subprocess, json, sys, numpy as np, cv2
from scipy.signal import stft, find_peaks
vid, evp, mp3, START = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
ev = json.load(open(evp)); FPS = 30
ROLLS = {"HAT", "OH", "TOM", "RIM", "TOMS"}
def isroll(v): return v.split("_")[0] in ROLLS
# master onsets per band
sr = 22050
y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", str(START), "-t", str(ev["dur"] + 0.5), "-i", mp3, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
fr, _, Z = stft(y, sr, nperseg=1024, noverlap=1024 - 128); S = np.abs(Z)
ons = {}
for nm, lo, hi in [("lo", 30, 200), ("mid", 200, 2000), ("hi", 5000, 11000)]:
    b = np.log1p(S[(fr >= lo) & (fr < hi)].sum(0)); fl = np.maximum(0, np.diff(b, prepend=b[0]))
    pk, _ = find_peaks(fl, height=fl.mean() + fl.std() * 1.2, distance=12)
    ons[nm] = pk * 128 / sr
def notemask(f):
    R, G, B = f[..., 2], f[..., 1], f[..., 0]
    return (((R > 110) & (R < 180) & (R - B > 30) & (R - B < 65) & (G < 160)) | ((np.maximum(np.maximum(R, G), B) < 150) & (R - B > 25) & (R > 45)))
def panel(f, layout):
    return f[800:] if layout == "face" else f
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", vid, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
frames = []
n = 0; shots = ev["shots"]
def shot_at(t):
    for s in shots:
        if s["t0"] <= t < s["t1"]: return s
while True:
    buf = p.stdout.read(1080 * 1920 * 3)
    if len(buf) < 1080 * 1920 * 3: break
    t = n / FPS; s = shot_at(t); n += 1
    if not s or not isroll(s["view"]): frames.append(None); continue
    f = panel(np.frombuffer(buf, np.uint8).reshape(1920, 1080, 3).astype(np.int16), s["layout"])
    H = f.shape[0]; reg = f[int(H * .1):int(H * .9):4]
    rg = (reg[..., 2] - reg[..., 1]).mean(0)
    if rg.max() < 45: frames.append(("noph", s)); continue
    ph = float(np.argmax(np.convolve(rg, np.ones(5) / 5, "same")))
    m = notemask(f).astype(np.uint8); m[:, max(0, int(ph) - 6):int(ph) + 7] = 0
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    bx = [(int(x0), int(y0 + h / 2)) for x0, y0, w, h, a in st[1:] if a > 100 and 15 <= h <= 520 and 6 <= w < 400]
    frames.append((ph, bx, s))
# tracks: per shot, chain boxes (same row, rel x decreasing a little each frame), fit, extrapolate the zero crossing
res = []
for i, fa in enumerate(frames):
    if not fa or fa[0] == "noph": continue
    for (xa, ya) in fa[1]:
        da = xa - fa[0]
        if not (8 < da < 160): continue
        # only start tracks at their first appearance in this lane
        pts = [(i, da)]; j = i; d = da
        while j + 1 < len(frames) and frames[j + 1] and frames[j + 1][0] != "noph" and frames[j + 1][2] is fa[2]:
            nb = [x - frames[j + 1][0] for (x, yy) in frames[j + 1][1] if abs(yy - ya) < 25 and -60 < (x - frames[j + 1][0]) - d <= 2]
            if not nb: break
            d = max(nb); j += 1
            if d <= 6: break
            pts.append((j, d))
        if len(pts) < 2: continue
        fx = np.array(pts, float); k1, k0 = np.polyfit(fx[:, 0], fx[:, 1], 1)
        if k1 >= -1: continue
        z = -k0 / k1
        if z - pts[-1][0] > 4 or z > len(frames) - 1: continue
        zf = int(z) + 1
        if zf >= len(frames) or not frames[zf] or frames[zf][0] == "noph" or frames[zf][2] is not fa[2]: continue
        res.append((round(z / FPS, 3), ya, fa[2]["view"]))
out = []
for r in sorted(res):
    if out and abs(r[0] - out[-1][0]) < 0.08 and abs(r[1] - out[-1][1]) < 40: continue
    out.append(r)
allons = np.sort(np.concatenate([ons["lo"], ons["mid"], ons["hi"]]))
bad = 0
for tc, yy, v in out:
    j = np.argmin(abs(allons - tc)); d = (tc - allons[j]) * 1000
    hi = ons["hi"][np.argmin(abs(ons["hi"] - tc))] if len(ons["hi"]) else 0
    flag = "" if abs(d) <= 35 else "  <-- OFF"
    bad += flag != ""
    print(f"{tc:7.3f}s y{yy:4d} {v:7s} nearest onset {d:+6.0f} ms (hi {1000 * (tc - hi):+6.0f}){flag}")
nop = sum(1 for f in frames if f and f[0] == "noph")
print(len(out), "crossings,", bad, "off;", nop, "roll frames without playhead")
