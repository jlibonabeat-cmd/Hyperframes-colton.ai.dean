"""Output frame at every visible note hit of every roll shot (ARRESTED) + 3 sample frames of every mixer/plugin/
channel-rack shot. sheets/<v>_hits.png, sheets/<v>_fixed.png"""
import json, cv2, numpy as np, sys
name = sys.argv[1]; R6 = 6 * 2.4
v = json.load(open("validate.json")); d = json.load(open(f"out/arr_{name}_events.json"))
for k in ("808", "HATPAT", "OH1"):
    v[k + "_R"] = {"cross": [(c[0] + R6,) + tuple(c[1:]) for c in v[k]["cross"] if 19.2 <= c[0] < 24.1]}
ST = d["start"]
class Lazy:
    def __init__(self, p): self.cap = cv2.VideoCapture(p); self.n = int(self.cap.get(7))
    def __len__(self): return self.n
    def __getitem__(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, f = self.cap.read(); return f
F = Lazy(f"out/arr_{name}.mp4")
def tile(f, lab, face):
    f = f[800:] if face else f
    f = cv2.resize(f, (216, int(f.shape[0] * 216 / 1080))); f = cv2.copyMakeBorder(f, 0, 384 - f.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(f, lab, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1); return f
T, X = [], []
for s in d["shots"]:
    vv = v.get(s["view"]); face = s["layout"] == "face"
    if s["view"] in ("MIXER", "MIXER_T", "PLUGIN", "COMP", "LAZER", "SCRATCH", "RESO"):
        for u in (0.1, 0.5, 0.9):
            n = int((s["t0"] + (s["t1"] - s["t0"]) * u) * 30); X.append(tile(F[n], f"{s['view']} n{n}", False))
    if not vv: continue
    ts = [c[0] - ST for c in vv["cross"] if s["t0"] <= c[0] - ST < s["t1"]]
    if len(ts) > 6: ts = ts[::max(1, len(ts) // 6)]           # dense hats: sample
    for tv in ts:
        n = int(round(tv * 30))
        if n < len(F): T.append(tile(F[n], f"{s['view']} {tv:.2f} n{n}", face))
def sheet(t, fn):
    rows = [np.hstack(t[i:i + 8] + [np.zeros_like(t[0])] * (8 - len(t[i:i + 8]))) for i in range(0, len(t), 8)]
    cv2.imwrite(fn, np.vstack(rows))
sheet(T, f"sheets/{name}_hits.png"); sheet(X, f"sheets/{name}_fixed.png"); print(name, len(T), "hit frames", len(X), "fixed frames", len(F), "frames")
