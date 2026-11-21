"""Hit sheet: the output frame at every note hit (picture crossing shifted into the view) inside each roll shot."""
import json, sys, cv2, numpy as np
VAL = json.load(open("validate_x.json"))
for v in sys.argv[1:]:
    d = json.load(open(f"out/agt_{v}_events.json")); c = cv2.VideoCapture(f"out/agt_{v}.mp4"); th = []
    for s in d["shots"]:
        if s["view"] not in VAL: continue
        for t, _ in [x for x in VAL[s["view"]]["cross"] if s["t0"] + 0.03 <= x[0] < s["t1"] - 0.02][:5]:
            n = int(round(t * 30)); c.set(1, n); ok, f = c.read()
            f = f[800:] if s["layout"] == "face" else f[400:1520]
            f = cv2.resize(f, (240, 249)); cv2.putText(f, f"{t:.2f} {s['view']}", (2, 14), 0, 0.42, (0, 0, 255), 1); th.append(f)
    rows = [np.hstack(th[i:i + 8] + [np.zeros_like(th[0])] * (8 - len(th[i:i + 8]))) for i in range(0, len(th), 8)]
    cv2.imwrite(f"sheets/{v}_hits.png", np.vstack(rows)); print(v, len(th), "hit frames")
