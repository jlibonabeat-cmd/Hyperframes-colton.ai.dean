"""Output frame at every visible note hit of every roll shot (CALI). sheets/<v>_hits.png"""
import json, subprocess, cv2, numpy as np, sys
name = sys.argv[1]
v = json.load(open("validate.json")); d = json.load(open(f"out/cali_{name}_events.json"))
v.setdefault("KICK", {"cross": []})["cross"] = [(40.45, 634, 3.4, 0), (45.484, 634, 8.459, 0)]
ST = d["start"]
cap = cv2.VideoCapture(f"out/cali_{name}.mp4"); frames = []
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
        cv2.putText(f, f"{s['view']} {tv:.2f} n{n}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1); tiles.append(f)
rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
cv2.imwrite(f"sheets/{name}_hits.png", np.vstack(rows)); print(name, len(tiles), "frames", len(frames))
