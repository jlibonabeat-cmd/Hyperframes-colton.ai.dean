"""Build one version of the instrument reel: plan (cuts on chord changes, angle order, take-grid-constrained best
take per shot, residual lag fix) + render (hard cuts, subtle zoom on long wide/front shots, master audio only)."""
import json, subprocess, sys, numpy as np, cv2
sys_argv = list(sys.argv)
sys.argv = ["x"]
exec(open("../scripts/cm_plan.py").read().split('if __name__=="__main__":')[0])
take_D = {"W": 9.62, "H": 8.205, "S": 7.98, "F": 12.79}
HB = BAR / 2
VERSIONS = {
    # name: (cuts in half bars over song 0..16 half bars = 0..21.33 s, angle order)
    "v2_original_zoom": ([0, 1, 2, 4, 6, 7, 8, 10, 12, 14, 16], ["W", "H", "S", "F", "H", "W", "S", "H", "W", "F"]),
    "v3_hands_first":   ([0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16], ["H", "S", "F", "H", "W", "S", "H", "F", "S", "W"]),
    "v4_slow_wide":     ([0, 2, 4, 8, 10, 12, 16], ["F", "H", "W", "S", "H", "W"]),
    "v5_every_chord":   (list(range(17)), ["S", "H", "F", "W", "H", "S", "F", "H", "W", "S", "H", "F", "S", "H", "W", "F"]),
}
ZOOM_ANGLES = {"W", "F"}                 # the far-away shots
ZOOM_MIN_LEN = 2.0                       # only on longer shots (s)
ZOOM_AMT = 0.08                          # subtle: 8 % over the shot
SR2 = 22050


def loadseg(p, ss, t):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(ss,0):.3f}", "-t", f"{t:.3f}", "-i", p, "-vn", "-ac", "1",
                                         "-ar", str(SR2), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)


def flux2(y, hop=55):
    w = 1024; n = (len(y) - w) // hop
    idx = np.arange(w)[None, :] + hop * np.arange(n)[:, None]
    X = np.log1p(np.abs(np.fft.rfft(y[idx] * np.hanning(w), axis=1))[:, 5:400] * 10)
    f = np.maximum(0, np.diff(X, axis=0)).sum(1); f = np.concatenate([[0], f])
    f = f - np.convolve(f, np.ones(80) / 80, mode="same"); return np.maximum(f, 0)


def residual(clip, c0, s0, L):
    a = flux2(loadseg(clip, c0 - 0.3, L + 0.6)); b = flux2(loadseg("track.mp3", s0, L))
    best = max(((np.corrcoef(a[k:k + len(b)], b)[0, 1], k) for k in range(0, len(a) - len(b))), key=lambda x: x[0])
    return best[1] * 55 / SR2 - 0.3


def plan_version(cuts, order):
    used = {k: [] for k in CLIPS}; plan = []
    for i in range(len(cuts) - 1):
        s0, s1 = cuts[i] * HB, cuts[i + 1] * HB
        opts = []
        for k in CLIPS:
            if plan and plan[-1]["angle"] == k:
                continue
            c0, sc = best_pos(k, s0, s1, used[k]); opts.append((sc + (0.3 if k == order[i] else 0), k, c0, sc))
        opts.sort(reverse=True)
        chosen = None
        for _, k, c0, sc in opts:               # best angle first; reject takes whose own audio doesn't match here
            for _try in range(4):
                if sc < -5:
                    break
                r = residual(CLIPS[k], c0, s0, s1 - s0)
                if abs(r) <= 0.1:
                    chosen = (k, c0 + r, sc); break
                print(f"    reject {k} @ {c0:.2f} (residual {r*1000:+.0f} ms)")
                c0, sc = best_pos(k, s0, s1, used[k] + [(c0, c0 + (s1 - s0))])
            if chosen:
                break
        k, c0, sc = chosen
        used[k].append((c0, c0 + (s1 - s0)))
        plan.append(dict(song0=s0, song1=s1, angle=k, clip=CLIPS[k], c0=round(c0, 3), score=round(sc, 3)))
    zi = 0
    for s in plan:
        if s["angle"] in ZOOM_ANGLES and s["song1"] - s["song0"] >= ZOOM_MIN_LEN:
            s["zoom"] = "in" if zi % 2 == 0 else "out"; zi += 1
        else:
            s["zoom"] = None
    return plan


def render(plan, out):
    FPS = 30; W, H = 1080, 1920
    NF = round(plan[-1]["song1"] * FPS)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-t", f"{NF/FPS+0.05:.3f}", "-i", "track.mp3", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "17",
                            "-preset", "medium", "-pix_fmt", "yuv420p", "-af", f"afade=t=out:st={NF/FPS-0.08:.3f}:d=0.08",
                            "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", out], stdin=subprocess.PIPE)
    n_prev = 0
    for i, s in enumerate(plan):
        n1 = round(s["song1"] * FPS) if i < len(plan) - 1 else NF
        k = n1 - n_prev; n_prev = n1
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{s['c0']:.3f}", "-i", s["clip"], "-frames:v", str(k), "-vf", "fps=30",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        last = None
        for j in range(k):
            b = p.stdout.read(W * H * 3)
            if len(b) == W * H * 3:
                last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
            img = last
            if s["zoom"]:
                u = j / max(1, k - 1); u = u * u * (3 - 2 * u)              # eased
                z = 1 + ZOOM_AMT * (u if s["zoom"] == "in" else 1 - u)
                M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.45]])   # centre slightly high (on Jack)
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
        p.stdout.close(); p.wait()
    enc.stdin.close(); enc.wait()


names = sys_argv[1:] or list(VERSIONS)
for name in names:
    cuts, order = VERSIONS[name]
    plan = plan_version(cuts, order)
    json.dump(plan, open(f"out/{name}_plan.json", "w"), indent=1)
    for s in plan:
        print(f"  {name} song {s['song0']:5.2f}-{s['song1']:5.2f} {s['angle']} @ {s['c0']:6.3f} score {s['score']:.2f} zoom {s['zoom']}")
    render(plan, f"out/catchme_{name}.mp4")
    print("rendered", name)
