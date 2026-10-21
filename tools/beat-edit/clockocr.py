import re, subprocess, sys
import numpy as np, cv2, pytesseract
def read_clock(f):
    crop = f[2:54, 893:1030]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
    inv = 255 - th
    inv = cv2.copyMakeBorder(inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    txt = pytesseract.image_to_string(inv, config="--psm 7 -c tessedit_char_whitelist=0123456789:").strip()
    m = re.fullmatch(r"(\d{1,3}):(\d{2}):(\d{2})", txt)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None, txt
if __name__ == "__main__":
    for c, t in [("kick", 5.4), ("kick", 15.0), ("kick", 21.5), ("kick", 27.0), ("snare1", 5.4), ("snare1", 9.4), ("snare2", 9.0), ("snare2", 12.4), ("s808", 5.95), ("s808", 15.3), ("hihat", 9.7), ("hihat", 14.7), ("hihat", 16.2), ("macro", 14.0), ("macro", 33.0)]:
        p = subprocess.run(["ffmpeg", "-nostats", "-v", "error", "-ss", str(t), "-i", f"{c}.mp4", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
        f = np.frombuffer(p.stdout, np.uint8).reshape(1080, 1920, 3)
        print(c, t, read_clock(f))
