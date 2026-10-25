"""Instrument reel: hard cuts on bar/half-bar lines, each shot = the clip position whose own audio matches the master
(constrained to the take's riff grid), residual-corrected; master audio only; subtle zoom only on long wide shots.
Footage is NOT colour-processed (Jack's request): just scaled."""
import json, subprocess, sys, numpy as np, cv2
SR = 22050; HOP = 110; DT = HOP / SR
BPM = 106.0; BAR = 240 / BPM; RIFF = 4 * BAR
T0 = 18.1132; NB = 8                                   # bars 0:18.11 -> 0:36.23
CLIPS = {"W": "c1.mp4", "K": "c2.mp4", "G": "c3.mp4"}  # wide room on keys, keys close-up, guitar mid shot
take_D = {"W": 17.69, "K": 17.72, "G": 17.91}
MAXC = {"W": 9.3, "K": 9.9, "G": 17.2}                 # usable clip end (G: stops playing after ~17.3 s)
ZOOM_ANGLES = {"W"}; ZOOM_MIN = 2.0; ZOOM_AMT = 0.08
VERS = {
    "main":       ([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8], ["W", "K", "G", "K", "W", "G", "K", "W", "G"]),
    "every_chord":([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
                   ["K", "G", "W", "K", "G", "K", "W", "G", "K", "G", "W", "K", "G", "K", "W", "G"]),
    "slow":       ([0, 2, 4, 6, 8], ["W", "K", "G", "W"]),
}
def load(p, ss=0, t=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(ss,0):.3f}", "-i", p] + (["-t", f"{t:.3f}"] if t else []) + ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)
def feats(y):
    w = 2048; n = (len(y) - w) // HOP
    idx = np.arange(w)[None, :] + HOP * np.arange(n)[:, None]
    X = np.abs(np.fft.rfft(y[idx] * np.hanning(w), axis=1))
    L = np.log1p(X[:, 5:400] * 10); f = np.concatenate([[0], np.maximum(0, np.diff(L, axis=0)).sum(1)])
    f = np.maximum(f - np.convolve(f, np.ones(40) / 40, mode="same"), 0)
    F = np.fft.rfftfreq(w, 1 / SR); m = (F > 70) & (F < 1800); pc = (np.round(12 * np.log2(F[m] / 261.63)) % 12).astype(int)
    C = np.stack([X[:, m][:, pc == k].sum(1) for k in range(12)], 1)
    C = np.array([np.convolve(C[:, k], np.ones(20) / 20, mode="same") for k in range(12)]).T
    return f, C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
fm, Cm = feats(load("track.mp3", 0, 60))
data = {k: feats(load(p)) for k, p in CLIPS.items()}
def best_pos(k, s0, s1, avoid):
    f, C = data[k]; a = int(round(s0 / DT)); L = int(round((s1 - s0) / DT))
    tz = (fm[a:a + L] - fm[a:a + L].mean()) / (fm[a:a + L].std() + 1e-9); tc = Cm[a:a + L]
    sc = np.full(len(f), -9.0)
    for c0 in range(0, len(f) - L):
        if c0 * DT + (s1 - s0) > MAXC[k]: break
        d = np.mod(s0 - c0 * DT - take_D[k] + RIFF / 2, RIFF) - RIFF / 2
        if abs(d) > 0.12: continue
        seg = f[c0:c0 + L]; z = (seg - seg.mean()) / (seg.std() + 1e-9)
        sc[c0] = (z * tz).mean() + 0.5 * (C[c0:c0 + L] * tc).sum(1).mean()
    for u0, u1 in avoid: sc[max(0, int(u0 / DT) - L):int(u1 / DT)] = -9
    k_ = int(np.argmax(sc)); return k_ * DT, float(sc[k_])
def flux2(y, hop=55):
    w = 1024; n = (len(y) - w) // hop
    idx = np.arange(w)[None, :] + hop * np.arange(n)[:, None]
    X = np.log1p(np.abs(np.fft.rfft(y[idx] * np.hanning(w), axis=1))[:, 5:400] * 10)
    f = np.concatenate([[0], np.maximum(0, np.diff(X, axis=0)).sum(1)]); f = f - np.convolve(f, np.ones(80) / 80, mode="same"); return np.maximum(f, 0)
def residual(clip, c0, s0, L, span=0.1):
    """fine lag (s) of clip audio vs master within +-span, plus its correlation."""
    a = flux2(load(clip, c0 - 0.3, L + 0.6)); b = flux2(load("track.mp3", s0, L))
    dt = 55 / SR; k0 = int(round(0.3 / dt)); ks = int(round(span / dt))
    best = max(((np.corrcoef(a[k:k + len(b)], b)[0, 1], k) for k in range(k0 - ks, k0 + ks + 1) if k >= 0 and k + len(b) <= len(a)), key=lambda x: x[0])
    return best[1] * dt - 0.3, best[0]


FINE = {}
def fine_offsets():
    """per take: median fine lag over half-bar windows with clear correlation (takes are continuous, no drift)."""
    for k, p in CLIPS.items():
        lags = []
        for s0 in np.arange(T0, T0 + NB * BAR - BAR, BAR / 2):
            c0 = s0 - take_D[k]
            while c0 + BAR > MAXC[k]: c0 -= RIFF
            while c0 < 0.05: c0 += RIFF
            if c0 + BAR > MAXC[k]: continue
            r, cr = residual(p, c0, s0, BAR)
            if cr >= 0.33: lags.append(r)
        FINE[k] = float(np.median(lags)); print(f"  take {k}: fine offset {FINE[k]*1000:+.1f} ms from {len(lags)} windows (spread {np.std(lags)*1000:.1f} ms)")
def plan(cuts, order):
    if not FINE: fine_offsets()
    used = {k: [] for k in CLIPS}; P = []
    for i in range(len(cuts) - 1):
        s0, s1 = T0 + cuts[i] * BAR, T0 + cuts[i + 1] * BAR; L = s1 - s0
        def cands(k):
            out = []
            for n in range(-3, 4):
                c0 = s0 - take_D[k] - n * RIFF + FINE[k]
                if c0 >= 0.05 and c0 + L <= MAXC[k]:
                    overlap = any(not (c0 + L <= u0 or c0 >= u1) for u0, u1 in used[k])
                    out.append((overlap, c0))
            return sorted(out)
        prefs = [order[i]] + [k for k in CLIPS if k != order[i]]
        chosen = None
        for k in prefs:
            if P and P[-1]["angle"] == k: continue
            c = cands(k)
            if c and (not c[0][0] or k == prefs[-1]):
                chosen = (k, c[0][1]); break
        if chosen is None:
            for k in prefs:
                if P and P[-1]["angle"] == k: continue
                c = cands(k)
                if c: chosen = (k, c[0][1]); break
        k, c0 = chosen; used[k].append((c0, c0 + L))
        P.append(dict(song0=s0, song1=s1, angle=k, clip=CLIPS[k], c0=round(c0, 3), score=0.0))
    zi = 0
    for s in P:
        s["zoom"] = None
        if s["angle"] in ZOOM_ANGLES and s["song1"] - s["song0"] >= ZOOM_MIN:
            s["zoom"] = "in" if zi % 2 == 0 else "out"; zi += 1
    return P
def render(P, out):
    FPS = 30; W, H = 1080, 1920; NF = round(NB * BAR * FPS)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-ss", f"{T0:.3f}", "-t", f"{NF/FPS+0.05:.3f}", "-i", "track.mp3", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "17",
                            "-preset", "medium", "-pix_fmt", "yuv420p", "-af", f"afade=t=out:st={NF/FPS-0.08:.3f}:d=0.08",
                            "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", out], stdin=subprocess.PIPE)
    n_prev = 0
    for i, s in enumerate(P):
        n1 = round((s["song1"] - T0) * FPS) if i < len(P) - 1 else NF
        k = n1 - n_prev; n_prev = n1
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{s['c0']:.3f}", "-i", s["clip"], "-frames:v", str(k), "-vf", "fps=30",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        last = None
        for j in range(k):
            b = p.stdout.read(W * H * 3)
            if len(b) == W * H * 3: last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
            img = last
            if s["zoom"]:
                u = j / max(1, k - 1); u = u * u * (3 - 2 * u); z = 1 + ZOOM_AMT * (u if s["zoom"] == "in" else 1 - u)
                M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.4]])
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
        p.stdout.close(); p.wait()
    enc.stdin.close(); enc.wait()
for name in (sys.argv[1:] or list(VERS)):
    P = plan(*VERS[name]); json.dump(P, open(f"out/{name}_plan.json", "w"), indent=1)
    for s in P:
        r, cr = residual(s["clip"], s["c0"], s["song0"], s["song1"] - s["song0"])
        print(f"  {name} {s['song0']:6.2f}-{s['song1']:6.2f} {s['angle']} @ {s['c0']:6.3f} zoom {s['zoom']} residual {r*1000:+.1f} ms corr {cr:.2f}")
    render(P, f"out/think_i_may_be_falling_{name}.mp4"); print("rendered", name)
