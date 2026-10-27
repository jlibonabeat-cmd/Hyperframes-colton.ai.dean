"""Output frame at every visible note hit of every roll shot (Dope Boy). sheets/<v>_hits.png"""
import json, cv2, numpy as np, sys
name = sys.argv[1]
BAR = 240 / 190; P8 = 8 * BAR
v = json.load(open("validate.json")); d = json.load(open(f"out/dope_{name}_events.json"))
v["SNARE8"] = {"cross": [(c[0] + P8,) + tuple(c[1:]) for c in v["SNARE"]["cross"]]}
v["808_8"] = {"cross": [(c[0] + P8,) + tuple(c[1:]) for c in v["808"]["cross"]]}
v["SNARE_W"] = v["SNARE"]; v["SNARE8_W"] = v["SNARE8"]
ST = d["start"]
cap = cv2.VideoCapture(f"out/dope_{name}.mp4"); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
tiles = []
for s in d["shots"]:
    vv = v.get(s["view"])
    if not vv: continue
    for c in vv["cross"]:
        tv = c[0] - ST
        if not (s["t0"] <= tv < s["t1"]): continue
        n = int(round(tv * 30))
        if n >= len(frames): continue
        f = frames[n].copy()
        if s["layout"] == "face": f = f[800:]
        f = cv2.resize(f, (216, int(f.shape[0] * 216 / 1080))); f = cv2.copyMakeBorder(f, 0, 384 - f.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(f, f"{s['view']} {tv:.2f} n{n}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1); tiles.append(f)
rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
cv2.imwrite(f"sheets/{name}_hits.png", np.vstack(rows)); print(name, len(tiles), "frames", len(frames))
