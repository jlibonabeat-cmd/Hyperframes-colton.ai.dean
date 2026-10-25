"""OCR FL clock every STEP s for each clip -> clockmap.json; then group into playback runs with offset D (song = clip + D)."""
import json, re, subprocess, sys
import numpy as np, cv2, pytesseract
BPM = 149.0
STEP = 0.2
def secs(B, S, T): return ((B - 1) + ((S - 1) + T / 24) / 16) * 4 * 60 / BPM
clips = sys.argv[1:]
out = {}
for clip in clips:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{clip}.mp4", "-vf", f"fps={1/STEP},crop=150:48:904:6",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 48, 150, 3)
    rows = []
    for i, f in enumerate(fr):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
        inv = cv2.copyMakeBorder(255 - th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        txt = pytesseract.image_to_string(inv, config="--psm 7 -c tessedit_char_whitelist=0123456789:").strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", txt)
        s = None
        if m and 1 <= int(m[2]) <= 16 and int(m[3]) < 24:
            s = secs(int(m[1]), int(m[2]), int(m[3]))
        rows.append([round(i * STEP, 2), s, txt])
    out[clip] = rows
    # runs: consecutive readings with same clip-song offset (within 0.06 s), moving forward
    runs = []
    for t, s, _ in rows:
        if s is None:
            continue
        d = s - t
        if runs and abs(d - np.median(runs[-1]["d"])) < 0.06 and t - runs[-1]["t"][-1] < 0.9:
            runs[-1]["t"].append(t); runs[-1]["d"].append(d)
        else:
            runs.append({"t": [t], "d": [d]})
    good = [(r["t"][0], r["t"][-1], float(np.median(r["d"]))) for r in runs if len(r["t"]) >= 4]
    print(clip, len(rows), "runs:", [(a, b, round(D, 3), round(a + D, 2), round(b + D, 2)) for a, b, D in good])
    out[clip + "_runs"] = good
json.dump(out, open("clockmap.json", "w"))
