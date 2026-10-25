import json, subprocess, cv2, numpy as np, sys
name = sys.argv[1]
v = json.load(open("validate.json")); d = json.load(open(f"out/glock_{name}_events.json"))
BAR = 240 / 149; ST = 0.032 + 4 * BAR; P8 = 8 * BAR; BEAT = BAR / 4
extra = {"HAT": [21.7 * BEAT, 53.7 * BEAT]}
tiles = []
for s in d["shots"]:
    ts = []
    vv = v.get(s["view"])
    if vv:
        for t, y, e in vv["cross"]:
            for sh in ((0, P8) if s["view"] in ("SNARE", "HAT") else (0,)):
                ts.append(t + sh - ST)
    ts += extra.get(s["view"], [])
    for tv in ts:
        if not (s["t0"] <= tv < s["t1"]): continue
        n = int(round(tv * 30))
        b = subprocess.run(["ffmpeg", "-v", "error", "-i", f"out/glock_{name}.mp4", "-vf", f"select='eq(n,{n})'", "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
        f = np.frombuffer(b, np.uint8).reshape(1920, 1080, 3).copy()
        if s["layout"] == "face": f = f[800:]
        f = cv2.resize(f, (216, int(f.shape[0] * 216 / 1080))); f = cv2.copyMakeBorder(f, 0, 384 - f.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(f, f"{s['view']} {tv:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); tiles.append(f)
rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
cv2.imwrite(f"sheets/{name}_hits.png", np.vstack(rows)); print(name, len(tiles))
