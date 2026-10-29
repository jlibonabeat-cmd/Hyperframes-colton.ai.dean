"""Frame-exact FL clock OCR (every EVERY-th native frame) -> per-run picture offset Dp (FL_time = clip_t + Dp).
Runs split on discontinuities / pauses / freezes. Writes clock2.json."""
import json, re, subprocess, sys
import numpy as np, cv2, pytesseract
from concurrent.futures import ProcessPoolExecutor
BPM = 100.0; BEAT = 60 / BPM; FPS = 60; EVERY = 3
def secs(B, S, T): return ((B - 1) + ((S - 1) + T / 24) / 16) * 4 * BEAT
def ocr(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(g, 110, 255, cv2.THRESH_BINARY)
    inv = cv2.copyMakeBorder(th if g.mean() > 128 else 255 - th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    txt = pytesseract.image_to_string(inv, config="--psm 7 -c tessedit_char_whitelist=0123456789:").strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):?(\d{2})", txt)
    if m and 1 <= int(m[2]) <= 16 and int(m[3]) < 24: return secs(int(m[1]), int(m[2]), int(m[3]))
    return None
def do(clip):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{clip}.mp4", "-vf", "crop=126:50:938:8", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 50, 126, 3)
    rows = [(k / FPS, ocr(fr[k])) for k in range(0, len(fr), EVERY)]
    pts = [(t, s) for t, s in rows if s is not None]
    runs = []
    for t, s in pts:
        d = s - t
        if runs and abs(d - np.median(runs[-1]["d"][-15:])) < 0.05 and t - runs[-1]["t"][-1] < 0.4:
            runs[-1]["t"].append(t); runs[-1]["d"].append(d); runs[-1]["s"].append(s)
        else:
            runs.append({"t": [t], "d": [d], "s": [s]})
    good = []
    for r in runs:
        if len(r["t"]) < 8: continue
        d = np.array(r["d"]); s = np.array(r["s"])
        # drop frozen readings (same clock value repeated) before taking the offset
        keep = np.concatenate([[True], np.diff(s) > 0])
        good.append((r["t"][0], r["t"][-1], float(np.median(d[keep])), float(np.percentile(d[keep], 25)), float(np.percentile(d[keep], 75)), len(r["t"])))
    return clip, rows, good
if __name__ == "__main__":
    out = {}
    with ProcessPoolExecutor(8) as ex:
        for clip, rows, good in ex.map(do, sys.argv[1:]):
            out[clip] = {"rows": rows, "runs": good}
            print(clip, [(round(a, 2), round(b, 2), round(D, 3), round(q3 - q1, 3), n) for a, b, D, q1, q3, n in good], flush=True)
    json.dump(out, open("clock2.json", "w"))
