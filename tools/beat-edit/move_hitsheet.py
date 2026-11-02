"""Output frame at every visible note hit of every roll shot (ARRESTED) + 3 sample frames of every mixer/plugin/
channel-rack shot. sheets/<v>_hits.png, sheets/<v>_fixed.png"""
import json, cv2, numpy as np, sys
name = sys.argv[1]
v = json.load(open("validate_x.json")); d = json.load(open(f"out/move_{name}_events.json"))
# heard moments: cluster each view's crossing estimates, snap the median to the master onset in that band (+-50 ms)
import subprocess
from scipy.signal import stft
_sr = 22050
_y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", str(d["start"]), "-t", str(d["dur"] + 0.5), "-i", "beat.mp3",
                                   "-ac", "1", "-ar", str(_sr), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
_fr, _, _Z = stft(_y, _sr, nperseg=1024, noverlap=1024 - 128); _S = np.abs(_Z); _T = np.arange(_S.shape[1]) * 128 / _sr
_F = {}
for _nm, _lo, _hi in [("sub", 30, 100), ("mid", 300, 3000)]:
    _b = np.log1p(_S[(_fr >= _lo) & (_fr < _hi)].sum(0)); _F[_nm] = np.maximum(0, np.diff(_b, prepend=_b[0]))
def heard_times(view, t0, t1):
    k = view.split("_")[0]; band = "sub" if k in ("SPINZ", "PHANTOM") else "mid"
    hs = sorted(c[0] - d["start"] for c in v[view]["cross"] if t0 <= c[0] - d["start"] < t1); cl = []
    for h in hs:
        if cl and h - cl[-1][-1] < 0.1: cl[-1].append(h)
        else: cl.append([h])
    out = []
    for c in cl:
        m = float(np.median(c)); i = (_T >= m - 0.05) & (_T <= m + 0.05)
        out.append(float(_T[i][np.argmax(_F[band][i])]) if i.any() else m)
    return out
ST = d["start"]
class Lazy:
    def __init__(self, p): self.cap = cv2.VideoCapture(p); self.n = int(self.cap.get(7))
    def __len__(self): return self.n
    def __getitem__(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, f = self.cap.read(); return f
F = Lazy(f"out/move_{name}.mp4")
def tile(f, lab, face):
    f = f[800:] if face else f
    f = cv2.resize(f, (216, int(f.shape[0] * 216 / 1080))); f = cv2.copyMakeBorder(f, 0, 384 - f.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(f, lab, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1); return f
T, X = [], []
EDGE = []
for s in d["shots"]:
    vv = v.get(s["view"]); face = s["layout"] == "face"
    if s["view"] in ("PLUGIN", "PLUGIN_F", "MACRO_W", "MACRO_T", "MACRO_K", "FUNK", "MIXER", "MIXER_T"):
        for u in (0.1, 0.5, 0.9):
            n = int((s["t0"] + (s["t1"] - s["t0"]) * u) * 30); X.append(tile(F[n], f"{s['view']} n{n}", False))
    if not vv: continue
    for tv in heard_times(s["view"], s["t0"], s["t1"]):
        n = int(tv * 30 + 0.5)
        fr_ = F[n] if n < len(F) else None
        if fr_ is not None:
            pn = fr_[800:] if face else fr_
            q = pn[100:-100:4].astype(int)
            g = ((q[..., 0] > 190) & (q[..., 2] > 150) & (q[..., 1] > 55) & (q[..., 1] < 130)).mean(0)
            rel = float(np.argmax(np.convolve(g, np.ones(5) / 5, "same"))) / pn.shape[1]
            EDGE.append((s["view"], round(tv, 2), round(rel, 2)))
        if n < len(F): T.append(tile(F[n], f"{s['view']} {tv:.2f} n{n}", face))
def sheet(t, fn):
    rows = [np.hstack(t[i:i + 8] + [np.zeros_like(t[0])] * (8 - len(t[i:i + 8]))) for i in range(0, len(t), 8)]
    cv2.imwrite(fn, np.vstack(rows))
sheet(T, f"sheets/{name}_hits.png"); sheet(X, f"sheets/{name}_fixed.png"); print(name, len(T), "hit frames", len(X), "fixed frames", len(F), "frames")
print("playhead position in panel at hits (flag <0.2 or >0.8):", [e for e in EDGE if not 0.2 <= e[2] <= 0.8] or "all inside")
