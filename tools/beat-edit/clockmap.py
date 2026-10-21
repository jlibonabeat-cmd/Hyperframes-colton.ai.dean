"""Map every clip frame to song time by OCR of FL's B:S:T clock. Output: clockmap.json {clip: [[tau, s_or_null], ...]}"""
import json, re, subprocess, sys
import numpy as np, cv2, pytesseract
STEP = 0.4
def secs(B, S, T): return ((B - 1) + ((S - 1) + T / 24) / 16) * 4 / 3
out = {}
for clip in ["kick", "snare1", "snare2", "s808", "hihat", "second8", "macro"]:
    raw = subprocess.run(["ffmpeg", "-nostats", "-v", "error", "-i", f"{clip}.mp4", "-vf", f"fps={1/STEP},crop=138:52:893:2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 52, 138, 3)
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
    print(clip, len(rows), "ok", sum(r[1] is not None for r in rows))
json.dump(out, open("clockmap.json", "w"))
