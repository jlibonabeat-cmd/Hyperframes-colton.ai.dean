"""Hit sheet: the output frame at every key-light note-on (FL song time + 0.692 = mp3, +13.333 for the _8 views) inside
each roll shot, cropped to the keyboard + playhead + the first notes. A note must sit on the playhead with its key lit."""
import json, sys, cv2, numpy as np
K8 = 13.333
for v in sys.argv[1:]:
    d = json.load(open(f"out/S2S_{v}_events.json")); c = cv2.VideoCapture(f"out/S2S_{v}.mp4"); th = []
    for s in d["shots"]:
        base = s["view"].split("_")[0].lower()
        if base not in ("bass", "chords", "counter", "serum"): continue
        sh = K8 if s["view"].endswith("_8") else 0.0
        ons = sorted({round(t + 0.692 + sh, 2) for t, y in json.load(open(f"{base}_keylit.json"))})
        for t in [x for x in ons if s["t0"] + 0.03 <= x < s["t1"] - 0.03][:6]:
            n = int(round(t * 30)); c.set(1, n); ok, f = c.read()
            f = f[806:, :540] if s["layout"] == "face" else f[:, :540]
            f = cv2.resize(f, (180, int(f.shape[0] * 180 / 540))); f = cv2.copyMakeBorder(f, 0, 640 - f.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(f, f"{t:.2f}", (2, 14), 0, 0.45, (0, 0, 255), 1); cv2.putText(f, s["view"][:9], (2, 30), 0, 0.45, (0, 0, 255), 1); th.append(f)
    rows = [np.hstack(th[i:i + 12] + [np.zeros_like(th[0])] * (12 - len(th[i:i + 12]))) for i in range(0, len(th), 12)]
    for k in range(0, len(rows), 2):
        cv2.imwrite(f"sheets/{v}_hits{k // 2}.png", np.vstack(rows[k:k + 2]))
    print(v, len(th), "hit frames", len(rows), "rows")
