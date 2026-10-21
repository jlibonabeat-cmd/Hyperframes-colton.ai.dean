"""v3: footage is time-locked to the song using the FL clock in each recording.

Every screen recording shows FL's own B:S:T transport clock. OCR of that clock (clockmap.py / segs2.json) gives, for each
playback segment of a recording, the exact song time it shows: song_time = clip_time + D. A shot at video time T (song time
START+T) therefore reads the clip at clip_time = START + T - D, at normal speed, so notes cross the playhead exactly when
they are heard. Crops follow the on-screen playhead. No text overlays.

Usage: python3 -I build_edit3.py out.mp4 [t_from t_to]
"""
import json
import math
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np
from scipy.signal import butter, sosfilt

import os
HERE = os.environ.get("BEAT_EDIT_DIR", os.getcwd())   # folder holding the clips, beat.mp3, segs2.json, hits.json, face/
W, H, FPS = 1080, 1920, 30
START = 21.3333                  # bar 17 of the song (the first drop)
DUR = 33.3333                    # 25 bars: first drop, breakdown, second drop
NF = int(round(DUR * FPS))
BEAT = 1.0 / 3.0
BAR = 4 * BEAT
MINT = (160, 230, 46)
SRC_W, SRC_H = 1920, 1080
FH = 800                        # face panel ~42% of the frame, wide framing (you sit smaller, like Glide / DTE / jsewell)

out_path = sys.argv[1]
t_a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t_b = float(sys.argv[3]) if len(sys.argv) > 3 else DUR
n_a, n_b = int(round(t_a * FPS)), min(NF, int(round(t_b * FPS)))


def fr(t):
    return int(round(t * FPS))


# ----------------------------------------------------------------------------- footage + song-time coverage
S2 = json.load(open(f"{HERE}/segs2.json"))


def D_of(clip, a, b):
    for (sa, sb, D, _) in S2[clip]:
        if abs(sa - a) < 0.15 and abs(sb - b) < 0.15:
            return D
    raise KeyError((clip, a, b))


def mk(clip, inst, kind, segs, cy, h, label_x=None):
    return dict(clip=clip, inst=inst, kind=kind, segs=segs, cy=cy, h=h)


# segs: (clip_a, clip_b, D) with margins to keep away from OBS windows / dialogs / view changes
V = {
    "KICK": mk("kick.mp4", "KICK", "roll", [(21.3, 27.0, D_of("kick", 21.2, 27.2))], 538, 760),
    "SNARE1": mk("snare1.mp4", "SNARE", "roll", [(6.5, 9.1, D_of("snare1", 6.4, 9.2))], 538, 760),
    "SNARE2": mk("snare2.mp4", "SNARE", "roll", [(3.3, 12.3, D_of("snare2", 3.2, 12.4))], 538, 760),
    "808A": mk("s808.mp4", "808", "roll", [(15.3, 21.5, D_of("s808", 15.2, 21.6))], 510, 780),
    "808B": mk("s808.mp4", "808", "roll", [(4.5, 7.1, D_of("s808", 4.4, 7.2))], 510, 780),
    "HAT": mk("hihat.mp4", "HAT", "roll", [(9.8, 13.9, D_of("hihat", 1.2, 14.0)), (14.9, 15.9, D_of("hihat", 14.8, 16.0))], 600, 800),
    "ARR_H": mk("hihat.mp4", "ARR", "play", [(1.7, 8.3, D_of("hihat", 1.2, 14.0))], 560, 880),
    "ARR_S": mk("second8.mp4", "ARR", "play", [(2.5, 4.3, D_of("second8", 2.4, 4.4)), (8.5, 11.1, D_of("second8", 8.4, 11.2)),
                                               (14.1, 29.5, D_of("second8", 14.0, 29.6))], 560, 880),
    "ARR_M": mk("macro.mp4", "ARR", "play", [(0.5, 31.9, 0.5 * (D_of("macro", 0.4, 24.4) + D_of("macro", 25.6, 32.0))),
                                             (32.9, 44.95, D_of("macro", 32.8, 42.4))], 560, 880),
}
PLAYHEAD_FALLBACK = {"macro.mp4": 984, "second8.mp4": 1352, "hihat.mp4": 952}


def coverage(view, s0, s1):
    """clip time for song time s0 if [s0, s1] sits inside one segment, else None -> (tau0, D)."""
    for (a, b, D) in V[view]["segs"]:
        if s0 - D >= a and s1 - D <= b:
            return s0 - D, D
    return None


FACE_CX = {"glideA": 0.69, "glideB": 0.69, "dteC": 0.40, "dteD": 0.38}
FACE_DUR = {k: v["b"] - v["a"] for k, v in json.load(open(f"{HERE}/face/info.json")).items()}
FACE_ORDER = ["glideA", "dteC", "glideB", "dteD", "glideB", "dteC", "dteD"]
FACE_START = {"glideA": 0.1, "glideB": 0.1, "dteC": 0.1, "dteD": 1.2}
face_state = {"i": 0, "t": FACE_START[FACE_ORDER[0]]}


def take_face(d):
    for _ in range(len(FACE_ORDER) + 2):
        nm = FACE_ORDER[face_state["i"] % len(FACE_ORDER)]
        if face_state["t"] + d + 0.15 <= FACE_DUR[nm]:
            s = face_state["t"]
            face_state["t"] += d + 0.05
            return nm, s
        face_state["i"] += 1
        face_state["t"] = FACE_START[FACE_ORDER[face_state["i"] % len(FACE_ORDER)]]
    return "glideA", 0.1


# ----------------------------------------------------------------------------- audio features
hits = json.load(open(f"{HERE}/hits.json"))                     # recomputed for this window
raw = subprocess.run(["ffmpeg", "-nostats", "-v", "error", "-ss", f"{START}", "-t", f"{DUR + 1}", "-i", f"{HERE}/beat.mp3",
                      "-vn", "-ac", "1", "-ar", "44100", "-f", "f32le", "-"], capture_output=True, check=True).stdout
y = np.frombuffer(raw, np.float32)
sub = np.abs(sosfilt(butter(4, [30, 90], btype="band", fs=44100, output="sos"), y))
sub_env = np.convolve(sub, np.ones(2205) / 2205, mode="same")[::441]
kt, ks = np.array(hits["kick"]["t"]), np.array(hits["kick"]["s"])
st_, ss_ = np.array(hits["snare"]["t"]), np.array(hits["snare"]["s"])
ht, hs = np.array(hits["hat"]["t"]), np.array(hits["hat"]["s"])


def feats(t0, t1):
    k = ks[(kt >= t0) & (kt < t1) & (ks >= 0.3)].sum()
    s = ss_[(st_ >= t0) & (st_ < t1) & (ss_ >= 0.45)].sum()
    h = ((ht >= t0) & (ht < t1)).sum() / max(t1 - t0, 0.1)
    b = sub_env[int(t0 * 100):max(int(t0 * 100) + 1, int(t1 * 100))].mean()
    return np.array([k, s, h, b])


grid = np.array([feats(t, t + 0.45) for t in np.arange(0.0, DUR - 0.5, 0.1)])
P90 = np.percentile(grid, 90, axis=0) + 1e-6
INSTS = ["KICK", "SNARE", "HAT", "808"]
WEIGHT = np.array([1.0, 0.9, 1.0, 1.0])


def feat_norm(t0, t1):
    return feats(max(0, t0 - 0.03), min(t1, t0 + 0.45)) / P90 * WEIGHT


def hat_roll(t):                                                  # dense hats in the last beat of a bar
    return ((ht >= t) & (ht < t + BEAT)).sum() >= 4


# ----------------------------------------------------------------------------- cut schedule (T seconds in the video)
onset_t = np.sort(np.concatenate([kt[ks >= 0.4], st_[ss_ >= 0.5]]))


def snap(t):
    if len(onset_t) == 0:
        return t
    d = np.abs(onset_t - t)
    i = d.argmin()
    return float(onset_t[i]) if d[i] <= 0.07 else t


NBARS = 25
bounds = [0.0]
for j in range(1, NBARS):
    drop = j < 8 or j >= 16
    if drop or j % 2 == 0:
        bounds.append(j * BAR)                                     # drops: every bar; breakdown: every 2 bars
bounds = sorted(set(round(snap(b), 3) for b in bounds))


def any_cov(t0, t1):
    s0, s1 = START + t0, min(START + t1 + 0.12, START + DUR - 0.2)
    return any(coverage(v, s0, s1) for v in V)


# footage edges (video time) where a recording starts/stops covering the song
edges = sorted(set(round(e - START, 2) for v in V.values() for (a_, b_, D_) in v["segs"] for e in (a_ + D_ + 0.08, b_ + D_ - 0.12)))
bounds.append(DUR)
fixed = [0.0]
for b in bounds[1:]:                                               # split a shot at a coverage edge if it would be uncovered
    a_ = fixed[-1]
    if not any_cov(a_, b):
        for e in edges:
            if a_ + 0.5 < e < b - 0.5 and (any_cov(a_, e) or any_cov(e, b)):
                fixed.append(e)
                a_ = e
                if any_cov(a_, b):
                    break
    fixed.append(b)
# merge consecutive shots with no footage (one calm face shot instead of several)
merged = [fixed[0]]
for i in range(1, len(fixed) - 1):
    if not any_cov(merged[-1], fixed[i]) and not any_cov(fixed[i], fixed[i + 1]):
        continue
    merged.append(fixed[i])
merged.append(DUR)
bounds = [b for k, b in enumerate(merged) if k == 0 or b - merged[k - 1] > 0.5]
if bounds[-1] != DUR:
    bounds[-1] = DUR

CLIP_INST_ORDER = ["KICK", "SNARE", "HAT", "808"]
PATTERN_VIEWS = {"KICK": ["KICK"], "SNARE": ["SNARE1", "SNARE2"], "HAT": ["HAT"], "808": ["808A", "808B"]}
ARR_VIEWS = ["ARR_S", "ARR_M", "ARR_H"]
rr = Counter()
last_view = None
last_inst = None
shots = []
face_ct = 0


def usable(view, s0, s1):
    return coverage(view, s0, min(s1 + 0.12, START + DUR - 0.2))


def choose(s0, s1, f, avoid):
    """pick the covered recording whose instrument is most active right now."""
    cands = []
    for inst, vs in PATTERN_VIEWS.items():
        sc = f[INSTS.index(inst)]
        for v in vs:
            if usable(v, s0, s1):
                cands.append((sc - (0.9 if v in avoid else 0.0) - (0.35 if inst == last_inst else 0.0), v))
    for v in ARR_VIEWS:
        if usable(v, s0, s1):
            cands.append(((0.95 if in_breakdown else 0.25) - (0.9 if v in avoid else 0.0), v))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    return cands[0][1], cands


for i in range(len(bounds) - 1):
    t0, t1 = bounds[i], bounds[i + 1]
    n0, n1 = fr(t0), fr(t1)
    if n1 <= n0:
        continue
    d = (n1 - n0) / FPS
    s0, s1 = START + n0 / FPS, START + n1 / FPS
    f = feat_norm(t0, t1)
    in_breakdown = BAR * 8 <= t0 < BAR * 16 - 1e-6
    k_shots = len(shots)
    pat = ["full", "face", "full", "face", "dual", "full", "face", "full"][k_shots % 8]
    if t0 < 0.01:
        pat = "full"
    if t0 >= DUR - 1.4 + 0.3:
        pat = "face" if k_shots % 2 == 0 else "full"
    if t0 > 0.01 and len(shots) >= 2 and all(x["layout"] != "face" for x in shots[-2:]):
        pat = "face"
    if pat == "face" and shots and shots[-1]["layout"] == "face":
        pat = "full"
    if t0 >= DUR - 2.8:
        pat = "face" if t0 >= DUR - 1.4 else "full"
    if d < 0.9 and pat == "dual":
        pat = "full"
    avoid = {last_view} if last_view else set()
    pick = choose(s0, s1, f, avoid)
    if pick is None:                                               # no recording covers this stretch -> you on top, nearest footage held below
        best = None
        for vv in ARR_VIEWS + ["HAT"]:
            for (a_, b_, D_) in V[vv]["segs"]:
                want = s0 - D_
                cl = min(max(want, a_), b_ - 0.05)
                if best is None or abs(want - cl) < best[0]:
                    best = (abs(want - cl), vv, cl)
        shot = dict(n0=n0, n1=n1, t0=t0, t1=t1, d=d, layout="face", idx=len(shots), f=[round(float(x), 2) for x in f],
                    views=[best[1]], insts=[V[best[1]]["inst"]], tau=[best[2]], face=take_face(d + 0.3), freeze=True)
        face_ct += 1
        last_view = best[1]
        shots.append(shot)
        continue
    va, cands = pick
    shot = dict(n0=n0, n1=n1, t0=t0, t1=t1, d=d, layout=pat, idx=len(shots), f=[round(float(x), 2) for x in f])
    if pat == "dual":
        others = [c for c in cands if c[1] != va and V[c[1]]["inst"] != V[va]["inst"]]
        if others:
            vb = others[0][1]
            shot["views"] = [va, vb]
        else:
            shot["layout"] = "full"
            shot["views"] = [va]
    else:
        shot["views"] = [va]
    if shot["layout"] == "face":
        shot["face"] = take_face(d + 0.3)
        face_ct += 1
    shot["insts"] = [V[v]["inst"] for v in shot["views"]]
    shot["tau"] = [coverage(v, s0, min(s1 + 0.12, START + DUR - 0.2))[0] for v in shot["views"]]
    last_view = va
    last_inst = V[va]["inst"]
    rr[va] += 1
    if shots and shots[-1]["layout"] == shot["layout"] and shots[-1]["views"] == shot["views"] and not shots[-1].get("freeze") and shot["layout"] != "face":
        shots[-1]["n1"], shots[-1]["t1"] = shot["n1"], shot["t1"]   # same view twice in a row -> one longer shot
        continue
    shots.append(shot)
shots[-1]["n1"] = NF

# ----------------------------------------------------------------------------- hit curves (per instrument)
Z = NF + 40
curve = {k: np.zeros(Z) for k in ("KICK", "SNARE", "HAT", "808", "ALL")}
shake = np.zeros(Z)
chroma_px = np.zeros(Z)
flash_a = np.zeros(Z)
cutpop = np.zeros(Z)


def add_decay(arr_, n, amp, tau, length=14):
    for k in range(length):
        if 0 <= n + k < len(arr_):
            arr_[n + k] += amp * math.exp(-k / (tau * FPS))


for t, s in zip(kt, ks):
    if t >= DUR - 0.02 or s < 0.25:
        continue
    a = 0.04 + 0.08 * s
    add_decay(curve["KICK"], fr(t), a, 0.10)
    add_decay(curve["808"], fr(t), a, 0.12)
    add_decay(curve["ALL"], fr(t), a, 0.10)
for t, s in zip(st_, ss_):
    if t >= DUR - 0.02 or s < 0.3:
        continue
    a = 0.025 + 0.06 * s
    add_decay(curve["SNARE"], fr(t), a, 0.08, 10)
    add_decay(curve["ALL"], fr(t), a * 0.8, 0.08, 10)
    if s >= 0.5:
        pass
for t, s in zip(ht, hs):
    if t >= DUR - 0.02 or s < 0.35:
        continue
    add_decay(curve["HAT"], fr(t), 0.02 + 0.05 * s, 0.05, 6)
for s in shots:
    pass
for drop_t in (0.0, BAR * 16):
    nd = fr(drop_t)
    for key in ("ALL", "KICK", "808"):
        add_decay(curve[key], nd, 0.10, 0.12, 14)
    add_decay(shake, nd, 5, 0.08, 8)
    for k, a in enumerate([0.55, 0.30, 0.12]):
        flash_a[nd + k] = max(flash_a[nd + k], a)
for j in (4, 8, 12, 20):
    n = fr(j * BAR)
    flash_a[n] = max(flash_a[n], 0.0)
for v in curve.values():
    v *= 0.35
    np.minimum(v, 0.04, out=v)
punch = np.zeros(Z)
shake_big = np.zeros(Z)
sel = []
thr = 0.33
thr_hi = 0.55
last_sel = -9.0
for t, s_ in zip(kt, ks):
    if t >= DUR - 1.2 or s_ < thr:
        continue
    bd = BAR * 8 <= t < BAR * 16
    if t - last_sel < (1.3 if bd else 0.95) or (bd and s_ < 0.6):
        continue
    sel.append((float(t), float(s_)))
    last_sel = t


def add_punch(n, amp, tau=3.5):
    for k in range(16):                                           # fast push-in over 2 frames, eased release
        if n + k < Z:
            punch[n + k] += amp * (min(k + 1, 2) / 2) * math.exp(-max(k - 1, 0) / tau)


for t, s_ in sel:
    big = s_ >= thr_hi
    add_punch(fr(t), 0.03 if big else 0.02)
    if big:
        pass
for t0_, a_ in ((0.0, 1.0), (BAR * 16, 1.0), (BAR * 12, 0.6), (BAR * 15 + 3 * BEAT, 0.7)):
    add_punch(fr(t0_), 0.015 * a_, 4.0)
# jsewell-style jolts: the footage snaps a few px sideways/up with motion blur on a hit, then settles
jx = np.zeros(Z)
jy = np.zeros(Z)
jb = np.zeros(Z)
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
PROF = [1.0, 0.6, 0.32, 0.15, 0.06]
for k_, (t, s_) in enumerate(sorted(sel)):
    amp = 110 if s_ > 5 else (85 if s_ >= thr_hi else 60)
    dx_, dy_ = DIRS[k_ % 4]
    n = fr(t)
    for q, p in enumerate(PROF):
        if n + q < Z:
            jx[n + q] += dx_ * amp * p
            jy[n + q] += dy_ * amp * p
            jb[n + q] = max(jb[n + q], [45, 25, 11, 0, 0][q])
print("jolt events", len(sel))
whip_idx = set()
for sh_ in shots:
    if sh_["idx"] > 0 and (sh_["idx"] % 3 == 1 or abs(sh_["t0"] - BAR * 8) < 0.1 or abs(sh_["t0"] - BAR * 16) < 0.1):
        whip_idx.add(sh_["n0"])
print("whip cuts", len(whip_idx))
events = {"cuts": [round(s["t0"], 3) for s in shots]}

# ----------------------------------------------------------------------------- drawing helpers


def blit_none():
    return None


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def find_playhead(frm, kind):
    """x (1920 space) of the thin full-height green playhead line, or None."""
    reg = frm[200:880:4].astype(np.int16)
    g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(axis=0)
    lo = 500 if kind == "roll" else 760
    cand = np.where(col[lo:] > 0.75)[0]
    return int(cand[0]) + lo if len(cand) else None


def warp(src, rect, size):
    x0, y0, cw, ch = rect
    sx, sy = size[0] / cw, size[1] / ch
    M = np.float32([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]])
    return cv2.warpAffine(src, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def held_rect(held, i, v, xph, zoom, aspect, split, lead):
    r = held[i]
    if r is not None and r[0] + 0.04 * r[2] <= xph <= r[0] + 0.96 * r[2]:
        return r
    r = view_rect(v, xph, zoom, aspect, split, lead)
    held[i] = r
    return r


def view_rect(v, xph, zoom, aspect, split, lead):
    base_h = v["h"] * (0.8 if split else 1.0)
    ch = base_h / zoom
    cw = ch * aspect
    xmin = 400 if v["kind"] == "roll" else 745
    x0 = xph - lead * cw
    x0 = min(max(x0, xmin), 1895 - cw)
    y0 = min(max(v["cy"] - ch / 2, 118), 1038 - ch)
    return x0, y0, cw, ch


def face_rect(nm, zoom, u, aspect):
    ch = 930 / zoom
    cw = min(ch * aspect, 1620)
    cx = FACE_CX[nm] * 1620 + (u - 0.5) * 40
    x0 = min(max(cx - cw / 2, 0), 1620 - cw)
    y0 = max(0, min(30 - (zoom - 1) * 80, 930 - ch))
    return x0, y0, cw, ch


def reader(path, ss, nframes, w, h_):
    cmd = ["ffmpeg", "-nostats", "-v", "error", "-ss", f"{ss:.3f}", "-i", path, "-vf", f"fps={FPS}",
           "-frames:v", str(nframes), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    last = np.zeros((h_, w, 3), np.uint8)
    size = w * h_ * 3
    for _ in range(nframes):
        buf = p.stdout.read(size)
        if len(buf) == size:
            last = np.frombuffer(buf, np.uint8).reshape(h_, w, 3)
        yield last
    p.stdout.close()
    p.wait()


def fisheye(img, k):
    u = (np.arange(W, dtype=np.float32) - W / 2) / (W / 2)
    v = (np.arange(H, dtype=np.float32) - H / 2) / (W / 2)
    uu, vv = np.meshgrid(u, v)
    f = 1.0 / (1.0 + k * (uu ** 2 + vv ** 2))
    return cv2.remap(img, ((uu * f) * (W / 2) + W / 2).astype(np.float32), ((vv * f) * (W / 2) + H / 2).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def chroma(img, px):
    if px < 1:
        return img
    px = int(px)
    o = img.copy()
    o[..., 0] = np.roll(img[..., 0], px, axis=1)
    o[..., 2] = np.roll(img[..., 2], -px, axis=1)
    return o


def shift(img, dx, dy):
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return img
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def tint(img, y0, y1, a):
    if a <= 0.01:
        return
    reg = img[y0:y1].astype(np.float32)
    img[y0:y1] = (reg * (1 - a) + np.array(MINT, np.float32) * a).astype(np.uint8)


# ----------------------------------------------------------------------------- render
enc = subprocess.Popen([
    "ffmpeg", "-nostats", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-ss", f"{START + n_a / FPS:.3f}", "-t", f"{(n_b - n_a) / FPS:.3f}", "-i", f"{HERE}/beat.mp3",
    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    "-profile:v", "high", "-movflags", "+faststart", "-af", f"afade=t=out:st={(n_b - n_a) / FPS - 0.12:.3f}:d=0.12", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path], stdin=subprocess.PIPE)

cut_frames = [s["n0"] for s in shots]
frames_done = 0
ph_log = []
for s in shots:
    f0, f1 = max(s["n0"], n_a), min(s["n1"], n_b)
    if f1 <= f0:
        continue
    off = (f0 - s["n0"]) / FPS
    k = f1 - f0
    if s.get("freeze"):
        fr0 = next(reader(f"{HERE}/{V[s['views'][0]]['clip']}", s["tau"][0], 1, SRC_W, SRC_H))
        readers = [iter([fr0] * k)]
    else:
        readers = [reader(f"{HERE}/{V[v]['clip']}", s["tau"][i] + off, k, SRC_W, SRC_H) for i, v in enumerate(s["views"])]
    freader = None
    if s["layout"] in ("face", "faceonly"):
        nm, fs0 = s["face"]
        freader = reader(f"{HERE}/face/{nm}.mp4", fs0 + off, k, 1620, 930)
    parity = s["idx"] % 2
    fh = FH
    last_x = [None, None]
    held = [None, None]
    for n in range(f0, f1):
        srcs = [next(r) for r in readers]
        fsrc = next(freader) if freader else None
        T = n / FPS
        u = (n - s["n0"]) / max(1, s["n1"] - s["n0"])
        zb = 1.0

        def pz(inst):
            return zb * (1.0 + curve["ALL" if inst == "ARR" else inst][n] + cutpop[n] + punch[n])

        def hit(inst):
            return min(1.0, curve["ALL" if inst == "ARR" else inst][n] / 0.09)

        def xph(i, vname, frm):
            v = V[vname]
            x = find_playhead(frm, v["kind"])
            if x is None:
                x = last_x[i] if last_x[i] is not None else PLAYHEAD_FALLBACK.get(v["clip"], 700)
            if last_x[i] is not None and abs(x - last_x[i]) < 150:
                x = 0.5 * x + 0.5 * last_x[i]          # light smoothing, snap on view jumps
            last_x[i] = x
            return x

        panels = []
        if s["layout"] == "full":
            vn = s["views"][0]
            v = V[vn]
            x = xph(0, vn, srcs[0])
            img = warp(srcs[0], held_rect(held, 0, v, x, pz(v["inst"]), W / H, False, 0.25), (W, H))
            panels.append((0, H, v["inst"]))
        elif s["layout"] == "faceonly":
            zf = 1.0 + punch[n]
            img = warp(fsrc, face_rect(s["face"][0], zf, 0.5, W / H), (W, H))
        elif s["layout"] == "dual":
            img = np.zeros((H, W, 3), np.uint8)
            for i, (vn, src) in enumerate(zip(s["views"], srcs)):
                v = V[vn]
                x = xph(i, vn, src)
                img[i * (H // 2):(i + 1) * (H // 2)] = warp(src, held_rect(held, i, v, x, pz(v["inst"]), W / (H / 2), True, 0.25), (W, H // 2))
                panels.append((i * (H // 2), (i + 1) * (H // 2), v["inst"]))
            img[H // 2 - 5:H // 2 + 5] = MINT
        else:
            img = np.zeros((H, W, 3), np.uint8)
            nm = s["face"][0]
            zf = 1.0 + punch[n]
            img[:fh] = warp(fsrc, face_rect(nm, zf, 0.5, W / fh), (W, fh))
            vn = s["views"][0]
            v = V[vn]
            bh = H - fh
            x = xph(0, vn, srcs[0])
            img[fh:] = warp(srcs[0], held_rect(held, 0, v, x, pz(v["inst"]), W / bh, True, 0.25), (W, bh))
            img[fh - 5:fh + 5] = MINT
            panels.append((fh, H, v["inst"]))
        for (py0, py1, inst) in panels:
            tint(img, py0, py1, 0.06 * hit(inst))
        # ---- effects
        d_cut = n - s["n0"]
        if abs(jx[n]) + abs(jy[n]) > 0.5:
            for (py0, py1, inst) in panels:
                py0, py1 = (py0 + 6 if py0 > 0 else 0), (py1 - 6 if py1 < H else H)   # keep the mint divider out of the jolt
                reg = np.ascontiguousarray(img[py0:py1])
                reg = cv2.warpAffine(reg, np.float32([[1, 0, jx[n]], [0, 1, jy[n]]]), (W, py1 - py0), borderMode=cv2.BORDER_REPLICATE)
                if jb[n] > 1:
                    ksz = (int(jb[n]), 1) if abs(jx[n]) > abs(jy[n]) else (1, int(jb[n]))
                    reg = cv2.blur(reg, ksz)
                img[py0:py1] = reg
        if (n + 1) in whip_idx:                                      # outgoing frame of a whip cut
            img = cv2.blur(shift(img, -50, 0), (61, 1))
        elif n in whip_idx:
            img = cv2.blur(shift(img, 70, 0), (61, 1))
        elif (n - 1) in whip_idx:
            img = cv2.blur(shift(img, 22, 0), (23, 1))
        fs = 0.0
        cp = 14 * fs
        img = chroma(img, cp)
        rng = np.random.default_rng(1000 + n)
        if fs > 0:
            amp = 10 * fs
            img = shift(img, rng.uniform(-amp, amp), rng.uniform(-amp, amp) * 0.6)
        if fs > 0:
            img = fisheye(img, 0.5 * fs)
            for _ in range(int(2 + 6 * fs)):
                if rng.random() < 0.25 + 0.7 * fs:
                    y0 = int(rng.integers(0, H - 140))
                    hb = int(rng.integers(14, 130))
                    sh = int(rng.integers(-170, 170) * (0.4 + fs))
                    img[y0:y0 + hb] = np.roll(img[y0:y0 + hb], sh, axis=1)
            for _ in range(int(1 + 4 * fs)):
                if rng.random() < 0.5 + 0.4 * fs:
                    y0 = int(rng.integers(0, H - 6))
                    img[y0:y0 + int(rng.integers(2, 6))] = 255
        fa = flash_a[n]
        if fs > 0 and (n % 4 == 0):
            fa = max(fa, 0.25 * fs)
        if fa > 0:
            img = cv2.addWeighted(img, 1 - fa, np.full_like(img, 255), fa, 0)
        if False:
            img = (img * (NF - 1 - n) / 4.0).astype(np.uint8)
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        frames_done += 1
enc.stdin.close()
enc.wait()
json.dump({"frames": frames_done, "start": START, "shots": [(s["t0"], s["t1"], s["layout"], s["views"], s.get("face"), s["tau"], s["f"]) for s in shots], **events},
          open(out_path.rsplit(".", 1)[0] + "_events.json", "w"))
print("frames", frames_done, "shots", len(shots), "face shots", face_ct)
print(Counter(v for s in shots for v in s["views"]))
