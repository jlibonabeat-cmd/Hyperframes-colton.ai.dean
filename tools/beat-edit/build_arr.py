"""ARRESTED (100 BPM, F minor) - 9:16 J Sewell beat edit, song bars 3-16 (0:04.88-0:38.48), drop at bar 9 (0:19.28),
808 + hats back at bar 15 (0:33.68, an exact repeat of bars 9-10 -> those views are reused 6 bars later).
Footage time-locked via frame-exact FL clock OCR (song = clip + Dp + B0); every recording's own audio matches the master
(corr 0.95-0.99) and every note hit lands on the song's 16th grid. Dark FL theme. Hats are follow-mode rolls (playhead
parked at x=690, notes scroll in). Smooth Operator cutaway reused from the Dope Boy session (not synced to this beat).
Usage (from the arr folder): VARIANT=A|B python3 -I build_arr.py out.mp4 [t_from t_to]
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.environ.get("BEAT_EDIT_DIR", os.getcwd())
W, H, FPS = 1080, 1920, 30
BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
B0 = 0.082                           # first downbeat in the mp3 (16th-grid phase)
START_J = 2                          # bar 3 (0:04.88)
NBARS = 14                           # bars 3-16 -> ends on the bar-17 downbeat (0:38.48)
START_BAR = START_J + 1
START = B0 + START_J * BAR
DUR = NBARS * BAR
NF = int(round(DUR * FPS))
FH = 800                             # face panel height (wide framing)
DIV = 3                              # half-height of the mint divider
MINT = (160, 230, 46)                # BGR
SRC_W, SRC_H = 1920, 1080
FACE_W, FACE_H = 1080, 800
FACE_SEGS = [(14.5, 27.6), (28.8, 33.2), (37.3, 41.6), (43.0, 48.8)]   # face_dope.mp4 (MVI_5558, 1458x1080 crop -> 1080x800): listening, eyes on the screen; lens looks 13-14, headphone adjust 6-12, hand gestures 28 / 42, head turns 33.5-37 and talking from 49 s excluded
REACT_T = 21.4                       # 'stank face' (21.4-23.2) -> pinned to REACT_J
FACE_Y0 = 0

out_path = sys.argv[1]
t_a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t_b = float(sys.argv[3]) if len(sys.argv) > 3 else DUR
n_a, n_b = int(round(t_a * FPS)), min(NF, int(round(t_b * FPS)))


def fr(t):
    return int(round(t * FPS))


# ----------------------------------------------------------------------------- views (clip, instrument, kind, playback segments)
# segments: (clip_a, clip_b, D) with song = clip + D (picture). Pattern-mode loops: D = Dp + B0 + 32 bars.
R6 = 6 * BAR                         # bars 15-16 repeat bars 9-10 (808 / hats / open hats)
ARR = dict(clip="macro.mp4", inst="ARR", kind="fixed", ph=982, xmin=600, segs=[(5.75, 46.4, -1.018)])
ROLL = dict(kind="roll", lo=457, xmin=440)
FOL = dict(kind="fixed", follow=True, ph=690, xmin=440)
V = {
    "MACRO": dict(ARR, cy=520, h=760, hs=700),
    "MACRO_T": dict(ARR, cy=300, h=420, hs=380),          # melody + clap rows
    "MACRO_D": dict(ARR, cy=640, h=520, hs=460),          # hats / 808 / bass rows
    "MACRO_P": dict(ARR, cy=560, h=600, hs=520),          # rims / percs rows
    "MACRO_W": dict(ARR, cy=560, h=900, hs=860),
    "CLAP": dict(ROLL, clip="clap.mp4", inst="CLAP", segs=[(4.35, 13.75, 6.826)], cy=633, h=520, hs=420, tz=1.5),
    "HAT": dict(FOL, clip="hat.mp4", inst="HAT", segs=[(5.0, 9.8, 9.451)], cy=673, h=560, hs=440, tz=1.3),
    "HATPAT": dict(FOL, clip="hatpat.mp4", inst="HAT", segs=[(5.4, 15.0, 13.869)], cy=438, h=560, hs=440, tz=1.25),
    "HATPAT_R": dict(FOL, clip="hatpat.mp4", inst="HAT", segs=[(5.4, 10.25, 13.869 + R6)], cy=438, h=560, hs=440, tz=1.25),
    "808": dict(ROLL, clip="s808.mp4", inst="808", segs=[(3.65, 13.0, 15.632)], cy=470, h=600, hs=500),
    "808_R": dict(ROLL, clip="s808.mp4", inst="808", segs=[(3.65, 8.5, 15.632 + R6)], cy=470, h=600, hs=500),
    "OH1": dict(ROLL, clip="oh1.mp4", inst="HAT", segs=[(2.5, 6.15, 16.757)], cy=560, h=520, hs=420, tz=1.4),
    "OH1_R": dict(ROLL, clip="oh1.mp4", inst="HAT", segs=[(2.5, 6.15, 16.757 + R6)], cy=560, h=520, hs=420, tz=1.4),
    "OH2": dict(ROLL, clip="oh2.mp4", inst="HAT", segs=[(2.55, 11.95, 16.719)], cy=560, h=520, hs=420, tz=1.4),
    "RIM": dict(ROLL, clip="rim.mp4", inst="SNARE", segs=[(2.3, 11.7, 26.551)], cy=560, h=520, hs=420, tz=1.4),
    # scratch fill: arrangement page view, playhead only on screen clip 1.9-3.2 (song 17.9-19.2); detect it in the empty
    # track below the waveform (the waveform hides it) and ride it
    "SCRATCH": dict(clip="scratch.mp4", inst="ARR", kind="roll", lo=700, xmin=600, ph_rows=(760, 990), pht=0.6, segs=[(1.85, 3.4, 16.02)], cy=610, h=700, hs=560, tz=1.15),
    # Reso bass = the PayPal [Bass] pattern (same notes), pattern mode; the pattern sits at song bars 5-8 -> D = Dp + B0 + 4 bars
    "RESO": dict(ROLL, clip="reso.mp4", inst="808", segs=[(4.65, 11.9, -4.7125 + 0.082 + 9.6)], cy=555, h=560, hs=440, tz=1.25),
    # channel rack: tight on the Zap 3 [Perc] (lazer) row + names, where he selects / puts the zap in (clip 5.3-6.0)
    "LAZER": dict(clip="lazer.mp4", inst="ARR", kind="fixed", segs=[(4.25, 8.25, 24.613)], cy=511, h=470, fx=895, xmin=0),
    "COMP": dict(clip="comp.mp4", inst="ARR", kind="fixed", segs=[(1.2, 6.05, 18.07)], cy=400, h=770, fx=768, xmin=0),
    # mixer: drum channels with moving meters (claps 2-3, open hats 4-5, rim/zap 6-7, HH 8, 808 9) / tight on HH + 808
    "MIXER": dict(clip="mixer.mp4", inst="ARR", kind="fixed", segs=[(1.15, 29.65, 8.569)], cy=560, h=810, fx=772, xmin=0),
    "MIXER_T": dict(clip="mixer.mp4", inst="ARR", kind="fixed", segs=[(1.15, 29.65, 8.569)], cy=330, h=430, fx=905, xmin=0),
    # Smooth Operator on the master, from the Dope Boy session (cutaway, window shown whole in the split panel)
    "PLUGIN": dict(clip="plugin.mp4", inst="ARR", kind="fixed", segs=[(0.85, 6.15, 27.9)], cy=450, h=820, fx=792, xmin=0),
}


PIC_LEAD = 0.02                      # OBS picture runs ~20 ms ahead of its own audio (measured on snare_main)


def coverage(view, s0, s1):
    for (a, b, D) in V[view]["segs"]:
        pl = V[view].get("pic", PIC_LEAD)
        if s0 - D - pl >= a - 0.08 and s1 - D - pl <= b:   # <=80 ms before a run = playhead parked at its start
            return s0 - D - pl
    return None


# ----------------------------------------------------------------------------- audio features
hits = json.load(open(f"{HERE}/hits.json"))
kt, ks = np.array(hits["kick"]["t"]), np.array(hits["kick"]["s"])
st_, ss_ = np.array(hits["snare"]["t"]), np.array(hits["snare"]["s"])
ht, hs = np.array(hits["hat"]["t"]), np.array(hits["hat"]["s"])


def act(inst, t0, t1):
    if inst == "HAT":
        return ((ht >= t0) & (ht < t1)).sum() / max(t1 - t0, 0.1) / 8.0
    if inst == "SNARE":
        return ss_[(st_ >= t0) & (st_ < t1) & (ss_ >= 0.4)].sum() / max(t1 - t0, 0.1) / 1.5
    if inst == "808":
        return ks[(kt >= t0) & (kt < t1) & (ks >= 0.3)].sum() / max(t1 - t0, 0.1) / 1.2
    return 0.35


# ----------------------------------------------------------------------------- shot plan
# cut points in bars relative to the start; 2-bar holds at the top, mid-drop and late drop
VARIANT = os.environ.get("VARIANT", "A")
# beats from 0:04.88 (bar 3). 0-8 melody only | 8-16 bars 5-6 | 16-24 bars 7-8 (hats, scratch fill) | 24 DROP (808) |
# 40-48 break (rim, lazer perc) | 48-56 bars 15-16 = repeat of bars 9-10. Visible hits (beats, validate.json):
# CLAP 11.02 13.05 15.05 17.04 19.03 21.05 (not drawn yet: 15.31 19.16 19.25) | HAT 16th notes 16.0-18.7 + 19.7-23.9 |
# 808 24.74 25.51 26.48 27.25 27.99 28.76 29.5 31.27 32.02 32.76 33.53 34.53 35.24 (not drawn yet: 26.85 29.05) |
# HATPAT 16ths 24.3-36 | OH1 27.47 | OH2 37.03 | RIM 45.74 46.23 53.76 54.24 | repeats +24 beats.
if VARIANT == "A":       # drum focus (CALI D style), Smooth Operator + lazer zap in the break
    B_ = [0, 4, 8, 11, 13.5, 16, 19.5, 21.75, 24, 26.7, 28.5, 30.5, 33, 35, 36.5, 38, 40, 44.5, 46.5, 48, 50.7, 52.5, 54.6, 56]
    L_ = ["full:MACRO_W", "face:MACRO_T", "full:RESO", "full:CLAP", "face:RESO", "full:HAT", "face:MACRO_W",
          "full:SCRATCH", "full:808", "full:OH1", "face:HATPAT", "full:808", "full:HATPAT", "face:808",
          "full:OH2", "full:MIXER_T", "face:LAZER", "full:RIM", "face:PLUGIN", "full:808_R",
          "face:OH1_R", "full:RIM", "face:MACRO_W"]
    REACT_J = 28.5 / 4
else:                    # B: variation with the mixer, the compressor on the melody and the lazer zap channel rack
    B_ = [0, 4, 8, 10.5, 13, 16, 18.5, 21.75, 24, 26.7, 29.3, 31.5, 33.5, 35.5, 37.5, 40, 42.5, 45, 46.6, 48, 50.7, 52.5, 54.6, 56]
    L_ = ["full:MACRO_W", "face:MACRO", "full:RESO", "face:CLAP", "full:RESO", "full:HAT", "face:MACRO_T",
          "full:SCRATCH", "full:HATPAT", "face:COMP", "full:808", "full:HATPAT", "face:808", "full:OH2",
          "full:MIXER_T", "face:LAZER", "full:MACRO_P", "full:RIM", "face:PLUGIN", "full:808_R", "face:OH1_R",
          "full:RIM", "face:MACRO"]
    REACT_J = 33.5 / 4
CUT_BARS = [b / 4 for b in B_[:-1]] + [NBARS]
FORCE = {b / 4: tuple(l.split(":")) for b, l in zip(B_, L_)}
TRANS_FORCE = {6: "zoom", 12: "zoom"}
DROP_J = 6                                   # 0:19.28 = the drop
SILENT_J = -1
shots = []
last_view, last_layout, since_face = None, None, 9
used = {}
for i in range(len(CUT_BARS) - 1):
    j0, j1 = CUT_BARS[i], CUT_BARS[i + 1]
    t0, t1 = j0 * BAR, j1 * BAR
    s0, s1 = START + t0, START + t1
    cands = []
    for vn, v in V.items():
        if coverage(vn, s0, s1 + 0.05) is None:
            continue
        if v.get("only") and j0 != 16:
            continue
        if vn == "SNARE_A":
            continue                             # mostly one note being re-drawn: never reads as in time
        sc = act(v["inst"], t0, t1) - 0.5 * used.get(v["clip"], 0)
        if v.get("only"):
            sc += 5.0                            # bars 33-34: wide arrangement shot, the breather before the outro
        if vn == last_view:
            sc -= 2.0
        if j0 in (SILENT_J, DROP_J) and vn == "MACRO":
            sc += 3.0                            # silent bar / drop downbeat: the whole arrangement stops / lights up
        cands.append((sc, vn))
    cands.sort(reverse=True)
    vn = FORCE[j0][1] if j0 in FORCE else cands[0][1]
    # layout: content first (808 / putting-in views need the wide split), then rhythm of face vs full
    if j0 in FORCE:
        lay = FORCE[j0][0]
    elif V[vn].get("split_only") or j0 == SILENT_J or j1 == NBARS:
        lay = "face"
    elif j0 == DROP_J or j1 == SILENT_J:
        lay = "full"
    elif last_layout == "face":
        lay = "full"
    elif since_face >= 2:
        lay = "face"
    else:
        lay = "full"
    shots.append(dict(idx=i, j0=j0, j1=j1, t0=t0, t1=t1, n0=fr(t0), n1=fr(t1), layout=lay, view=vn,
                      tau=coverage(vn, s0, s1 + 0.05), cands=[c[1] for c in cands]))
    last_view, last_layout = vn, lay
    used[V[vn]["clip"]] = used.get(V[vn]["clip"], 0) + 1
    since_face = 0 if lay == "face" else since_face + 1
shots[-1]["n1"] = NF
VAL = json.load(open(f"{HERE}/validate.json")) if os.path.exists(f"{HERE}/validate.json") else {}
for _k in ("808", "HATPAT", "OH1"):            # bars 15-16 repeat bars 9-10 (song 19.28-24.08)
    if _k in VAL:
        VAL[_k + "_R"] = {"cross": [(c[0] + R6,) + tuple(c[1:]) for c in VAL[_k]["cross"] if 19.2 <= c[0] < 24.1]}
for x in shots:
    vv = VAL.get(x["view"])
    if not vv:
        continue
    s0_, s1_ = START + x["t0"], START + x["t1"]
    ys = [c[1] for c in vv["cross"] if s0_ <= c[0] < s1_]
    x["n_cross"] = len(ys)
    if ys:
        x["cy"] = 0.5 * (min(ys) + max(ys))
bad = [(round(x["t0"], 2), x["view"]) for x in shots if x["tau"] is None]
if bad:
    raise SystemExit(f"no footage coverage for {bad}")
fq = [list(x) for x in FACE_SEGS]
for s_ in shots:
    if s_["j0"] == REACT_J:
        s_["face_t"] = REACT_T
        for seg in fq:
            if seg[0] <= REACT_T < seg[1]:
                fq.append([REACT_T + (s_["n1"] - s_["n0"]) / FPS + 0.1, seg[1]]); seg[1] = REACT_T
for s_ in sorted([x for x in shots if x["layout"] == "face" and "face_t" not in x], key=lambda x: -(x["n1"] - x["n0"])):
    need = (s_["n1"] - s_["n0"]) / FPS + 0.1
    fits = [seg for seg in fq if seg[1] - seg[0] >= need]
    if fits:
        seg = min(fits, key=lambda g: (g[1] - g[0]) - need)       # best fit
        s_["face_t"] = seg[0]
        seg[0] += need
    else:
        raise SystemExit(f"not enough clean face footage for shot at {s_['t0']:.2f}")

# zoom moves (never below 1.0 so the crop never runs out of picture):
#   impact  : opens zoomed in and snaps out (1.25 -> 1.0 in ~10 frames, zoom-blurred), then a slow push
#   mixer   : slow push in + a punch on every bar downbeat (the meters alone are dull)
#   push/pull: slow 7% zoom in or out across the shot (occasional, on DAW-only shots)
TILT_SIGN = [1, -1]
nt = 0
pp = 0
for s in shots:
    v = V[s["view"]]
    s["theta"] = 0.0
    if s["idx"] == 0:
        s["zm"] = "impact"
    elif s["view"] in ("MIXER", "MIXER_T", "PLUGIN", "COMP"):
        s["zm"] = "mixer"
    elif s["layout"] == "full" and (v["kind"] == "roll" or v.get("follow")):
        s["zm"] = "track"                       # tight macro, camera follows the playhead
    elif s["layout"] == "full" and s["view"] == "MACRO_W":
        s["zm"] = "push"
    elif s["layout"] == "full" and s["view"] in ("MACRO", "MACRO_T", "MACRO_D", "MACRO_P") and s["j0"] not in (DROP_J,):
        s["zm"] = "tilt"                        # tilted playlist, slow rotate + push (J Sewell)
        s["theta"] = TILT_SIGN[nt % 2]
        nt += 1
    elif s["layout"] == "face":
        s["zm"] = "push" if pp % 2 == 0 else "pull"
        pp += 1
    else:
        s["zm"] = None


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def zoom_at(s, n):
    u = (n - s["n0"]) / max(1, s["n1"] - s["n0"])
    k = n - s["n0"]
    zm = s.get("zm")
    if zm == "impact":
        return 1.0 + 0.25 * (1 - smooth(k / 10.0)) ** 2 + 0.06 * smooth(u)
    if zm == "mixer":
        z = 1.0 + 0.14 * smooth(u)
        for j in range(int(math.ceil(s["j0"] - 1e-6)), int(math.ceil(s["j1"] - 1e-6))):
            d = n - fr(j * BAR)
            if 0 <= d < 12:
                z += 0.07 * math.exp(-d / 3.5)
        return z
    if zm == "track":
        if V[s["view"]]["inst"] == "808":
            return 1.12 + 0.10 * smooth(u)          # 808 focus: wider so the whole 808 phrase reads, slow push
        return V[s["view"]].get("tz", 1.55) + 0.08 * smooth(u)
    if zm == "tilt":
        return 1.26 + 0.10 * smooth(u)
    if zm == "push":
        return 1.0 + 0.07 * smooth(u)
    if zm == "pull":
        return 1.07 - 0.07 * smooth(u)
    return 1.0


def theta_at(s, n):
    if s.get("zm") != "tilt":
        return 0.0
    u = (n - s["n0"]) / max(1, s["n1"] - s["n0"])
    return s["theta"] * (7.0 - 4.0 * smooth(u))


def zoomed(r, z, v, x):
    x0, y0, cw, ch = r
    if z <= 1.0005:
        return r
    if "ph" in v:
        fx = v["ph"]
    elif "fxl" in v:
        fx = x0 + 0.08 * cw
    elif x is not None and x0 <= x <= x0 + cw:
        fx = x
    else:
        fx = x0 + cw / 2
    fy = y0 + ch / 2
    return (fx - (fx - x0) / z, fy - (fy - y0) / z, cw / z, ch / z)


def fit_panel(src, v, size):
    """whole pattern visible (you can see every note being placed): scale the pattern area to the panel width,
    fill the rest with a blurred, darkened copy."""
    x0, y0, cw, ch = v["fit"]
    w = size[0]
    h = int(round(ch * w / cw))
    fg = cv2.resize(src[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(src[100:1040, x0:x0 + cw], (w, size[1]), interpolation=cv2.INTER_LINEAR)
    bg = (cv2.GaussianBlur(bg, (0, 0), 25) * 0.45).astype(np.uint8)
    top = (size[1] - h) // 2
    bg[top:top + h] = fg
    return bg


def cam(src, r, z, th, v, x, size):
    """crop rect r zoomed by z (about the play position) and rotated by th degrees, kept inside r."""
    zx0, zy0, zw, zh = zoomed(r, z, v, x)
    cx, cy = zx0 + zw / 2, zy0 + zh / 2
    if abs(th) > 0.01:
        c, sn = math.cos(math.radians(th)), abs(math.sin(math.radians(th)))
        bw, bh = zw * c + zh * sn, zw * sn + zh * c
        x0, y0, cw, ch = r
        cx = min(max(cx, x0 + bw / 2), x0 + cw - bw / 2)
        cy = min(max(cy, y0 + bh / 2), y0 + ch - bh / 2)
    sc = size[0] / zw
    a = math.radians(th)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * sc
    t = np.array([size[0] / 2, size[1] / 2]) - R @ np.array([cx, cy])
    M = np.float32(np.hstack([R, t[:, None]]))
    return cv2.warpAffine(src, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def panel(src, r, size, v, x, z, zprev, th=0.0, thprev=0.0):
    """DAW panel; if the camera jumped this frame (zoom/rotation), average the in-between positions (motion blur)."""
    if abs(z - zprev) < 0.012 and abs(th - thprev) < 0.4:
        return cam(src, r, z, th, v, x, size)
    acc = np.zeros((size[1], size[0], 3), np.float32)
    steps = 7
    for i in range(steps):
        u = i / (steps - 1)
        acc += cam(src, r, zprev + (z - zprev) * u, thprev + (th - thprev) * u, v, x, size)
    return (acc / steps).astype(np.uint8)


def zoom_blur(img, z0, z1, samples=8):
    """radial zoom blur of the whole frame: average of scalings about the centre from z0 to z1."""
    h_, w_ = img.shape[:2]
    acc = np.zeros(img.shape, np.float32)
    for i in range(samples):
        zz = z0 + (z1 - z0) * i / (samples - 1)
        M = np.float32([[zz, 0, (1 - zz) * w_ / 2], [0, zz, (1 - zz) * h_ / 2]])
        acc += cv2.warpAffine(img, M, (w_, h_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (acc / samples).astype(np.uint8)


# ----------------------------------------------------------------------------- motion events
# whips: drop + every third cut (not into the silent bar, not into the last shot)
TRANS_CYCLE = ["whipH", "zoom", "hard", "whipV", "whipH", "hard", "zoom", "whipV"]
trans = {}
k_ = 0
for s in shots:
    if s["idx"] == 0 or s["j0"] == SILENT_J:
        continue                                   # into the silent bar: hard cut on the drop-out
    if s["j0"] in TRANS_FORCE:
        trans[s["n0"]] = TRANS_FORCE[s["j0"]]
        continue
    trans[s["n0"]] = TRANS_CYCLE[k_ % len(TRANS_CYCLE)]
    k_ += 1
# never smear a note hit: a cut within 3.5 frames of a hit visible in the outgoing or incoming shot becomes a hard cut
def view_hits(vn):
    return [c[0] - START for c in VAL.get(vn, {}).get("cross", [])]
for i_, s_ in enumerate(shots):
    n_ = s_["n0"]
    if n_ not in trans:
        continue
    near = view_hits(s_["view"]) + (view_hits(shots[i_ - 1]["view"]) if i_ > 0 else [])
    if s_["j0"] not in TRANS_FORCE and any(abs(n_ / FPS - h) < 3.5 / FPS for h in near):
        trans[n_] = "hard"
whip_in = {n for n, t in trans.items() if t != "hard"}
# jolts: rare — the drop downbeat and the 3 strongest 808 hits in the drop that are not on a cut
drop_t = DROP_J * BAR
cand = [(s, t) for t, s in zip(kt, ks) if t > drop_t + 0.5 and t < DUR - 2 * BAR and s >= 0.45]
cand.sort(reverse=True)
jolts = []   # the drop itself gets a whip + soft flash, not a jolt
for s, t in cand:
    sh_ = [x for x in shots if x["t0"] <= t < x["t1"]]
    near_hit = (not sh_) or any(abs(t - h) < 0.25 for h in view_hits(sh_[0]["view"]))
    if not near_hit and all(abs(t - u) > 2.0 for u, _ in jolts) and all(abs(fr(t) - w) > 6 for w in whip_in) and len(jolts) < 2:
        jolts.append((float(t), 0.8))
jolts.sort()
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
PROF = [1.0, 0.5, 0.2, 0.06, 0.0]           # offset (as fraction of amp) at the end of each frame
jx = np.zeros(NF + 10)
jy = np.zeros(NF + 10)
for k, (t, a) in enumerate(jolts):
    n = fr(t)
    dx, dy = DIRS[k % 4]
    amp = 90 * a
    for q, p in enumerate(PROF):
        if n + q < len(jx):
            jx[n + q] = dx * amp * p
            jy[n + q] = dy * amp * p
SHAKE = os.environ.get("SHAKE", "1") == "1"
shx = np.zeros(NF + 12); shy = np.zeros(NF + 12)
if SHAKE:
    rs = np.random.default_rng(149)
    for n0_, kind_ in trans.items():
        if kind_ == "hard":
            continue
        amp = 26.0 if kind_ == "zoom" else 18.0
        ang = rs.uniform(0, 2 * np.pi)
        for k in range(1, 8):                     # starts right after the transition lands
            a_ = amp * math.exp(-(k - 1) / 1.8) * (1 if k % 2 else -1)
            ang += rs.uniform(-0.9, 0.9)
            if n0_ + k < len(shx):
                shx[n0_ + k] = a_ * math.cos(ang); shy[n0_ + k] = a_ * math.sin(ang) * 0.8
flash = np.zeros(NF + 10)
for k, a in enumerate([0.35, 0.18, 0.07]):
    flash[fr(drop_t) + k] = a

# ----------------------------------------------------------------------------- helpers


def reader(path, ss, nframes, w, h_):
    cmd = ["ffmpeg", "-nostats", "-v", "error", "-ss", f"{max(ss, 0):.3f}", "-i", path, "-vf", f"fps={FPS}",
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


def find_playhead(frm, kind, lo=None, light=False, rows=None, pht=0.75):
    if kind == "fixed":
        return None
    if rows:                                 # playhead hidden behind a waveform: look in an empty track instead
        reg = frm[rows[0]:rows[1]:3].astype(np.int16)
        g = (reg[..., 1] > 70) & (reg[..., 1] - reg[..., 2] > 35) & (reg[..., 1] - reg[..., 0] > 25)
        c = np.where(g.mean(axis=0)[lo:1890] > pht)[0]
        return int(c[0]) + lo if len(c) else None
    if light:                                # FL light theme: brown-orange 1-px playhead core
        reg = frm[210:870:5].astype(np.int16)
        g = (np.abs(reg[..., 0] - 91) < 22) & (np.abs(reg[..., 1] - 140) < 30) & (np.abs(reg[..., 2] - 185) < 28)
        c = np.where(g.mean(axis=0)[lo:1900] > 0.8)[0]
        return int(c[0]) + lo if len(c) else None
    reg = frm[200:880:4].astype(np.int16)
    g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(axis=0)
    lo = lo if lo is not None else (500 if kind == "roll" else 760)
    c = np.where(col[lo:1890] > 0.75)[0]
    return int(c[0]) + lo if len(c) else None


def warp(src, rect, size):
    x0, y0, cw, ch = rect
    sx, sy = size[0] / cw, size[1] / ch
    M = np.float32([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]])
    return cv2.warpAffine(src, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def view_rect(v, x, aspect, lead=0.3):
    ch = v.get("h", 900) if aspect < 0.7 else (v["h"] if v["kind"] == "fixed" and "fx" in v else v.get("hs", 780))
    cw = ch * aspect
    xmin = v.get("xmin", 500 if v["kind"] == "roll" else 745)
    if v["kind"] == "fixed":
        if "ph" in v:                      # playlist in follow mode: playhead parked at x=ph, the song scrolls under it
            x0 = v["ph"] - 0.35 * cw
        elif "fxl" in v:                   # piano roll in smooth-follow: notes stream left and play at the keyboard edge
            x0 = v["fxl"]
        else:
            x0 = v["fx"] - cw / 2
    else:
        x0 = x - lead * cw
    x0 = min(max(x0, xmin), 1888 - cw)
    y0 = min(max(v["cy"] - ch / 2, 100), 1040 - ch) if ch < 940 else 100
    return x0, y0, cw, ch


def held(state, v, x, aspect):
    r = state.get("r")
    if v["kind"] == "fixed" or x is None:
        if r is None:
            r = view_rect(v, x if x is not None else 1100, aspect)
        state["r"] = r
        return r
    if r is not None and r[0] + 0.05 * r[2] <= x <= r[0] + 0.95 * r[2]:
        return r
    r = view_rect(v, x, aspect)
    state["r"] = r
    return r


def draw_playhead(img, y0, y1, r, v, size):
    """FL-style playhead (dark-green glow + bright 1px core) at the view's real play position."""
    x0, _, cw, _ = r
    sx = size[0] / cw
    X = int(round((v["ph"] - x0) * sx))
    g = max(3, int(round(4 * sx)))
    if 0 <= X < size[0]:
        a = max(0, X - g)
        reg = img[y0:y1, a:X].astype(np.float32)
        if v.get("light"):
            img[y0:y1, a:X] = (reg * 0.4 + np.array((150, 205, 255), np.float32) * 0.6).astype(np.uint8)
            img[y0:y1, X:min(size[0], X + max(2, int(sx)))] = (60, 125, 215)
        else:
            img[y0:y1, a:X] = (reg * 0.35 + np.array((34, 89, 0), np.float32) * 0.65).astype(np.uint8)
            img[y0:y1, X:min(size[0], X + max(2, int(sx)))] = (133, 189, 80)


def shift_img(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def motion_blur_path(img, x0, y0, x1, y1, samples=12):
    """average of the image shifted along the path (x0,y0)->(x1,y1): real directional motion blur."""
    if abs(x1 - x0) + abs(y1 - y0) < 1.5:
        return shift_img(img, x1, y1)
    acc = np.zeros(img.shape, np.float32)
    for k in range(samples):
        u = k / (samples - 1)
        acc += shift_img(img, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
    return (acc / samples).astype(np.uint8)


# ----------------------------------------------------------------------------- render
enc = subprocess.Popen([
    "ffmpeg", "-nostats", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-ss", f"{START + n_a / FPS:.3f}", "-t", f"{(n_b - n_a) / FPS:.3f}", "-i", f"{HERE}/beat.mp3",
    "-map", "0:v", "-map", "1:a", "-af", f"afade=t=out:st={(n_b - n_a) / FPS - 0.12:.3f}:d=0.12",
    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high",
    "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path], stdin=subprocess.PIPE)

whip_out = {n - 1 for n in whip_in}
frames_done = 0
for s in shots:
    f0, f1 = max(s["n0"], n_a), min(s["n1"], n_b)
    if f1 <= f0:
        continue
    off = (f0 - s["n0"]) / FPS
    k = f1 - f0
    v = dict(V[s["view"]], **({"cy": s["cy"]} if "cy" in s else {}))
    rd = reader(f"{HERE}/{v['clip']}", s["tau"] + off, k, SRC_W, SRC_H)
    frd = reader(f"{HERE}/face_dope.mp4", s["face_t"] + off, k, FACE_W, FACE_H) if s["layout"] == "face" else None
    st = {}
    lastx = [None]
    for n in range(f0, f1):
        src = next(rd)
        x = find_playhead(src, v["kind"], v.get("lo"), v.get("light", False), v.get("ph_rows"), v.get("pht", 0.75))
        if x is not None and lastx[0] is not None and abs(x - lastx[0]) < 150:
            x = 0.5 * x + 0.5 * lastx[0]
        if x is not None:
            lastx[0] = x
        else:
            x = lastx[0]
        z = zoom_at(s, n)
        zp = zoom_at(s, n - 1) if n > s["n0"] else (1.45 if s.get("zm") == "impact" else z)
        th = theta_at(s, n)
        thp = theta_at(s, n - 1) if n > s["n0"] else th
        if s.get("zm") == "track" and x is not None:
            xs_ = st.get("xs")
            xs_ = x if xs_ is None else 0.7 * xs_ + 0.3 * x
            st["xs"] = xs_
            st["r"] = None
        if s["layout"] == "full":
            if s.get("zm") == "track" and v["kind"] == "roll" and x is not None:
                r = view_rect(v, st["xs"], W / H, lead=0.42)     # camera rides the playhead
            else:
                r = held(st, v, x, W / H)
            img = panel(src, r, (W, H), v, x, z, zp, th, thp)
            if v.get("draw_ph"):
                draw_playhead(img, 0, H, zoomed(r, z, v, x), v, (W, H))
            dy0, dy1 = 0, H
        else:
            img = np.empty((H, W, 3), np.uint8)
            fsrc = next(frd)
            img[:FH] = fsrc[FACE_Y0:FACE_Y0 + FH]
            bh = H - FH
            r = held(st, v, x, W / bh)
            img[FH:] = panel(src, r, (W, bh), v, x, z, zp)
            if v.get("draw_ph"):
                draw_playhead(img, FH, H, zoomed(r, z, v, x), v, (W, bh))
            dy0, dy1 = FH + DIV, H
        # jolt: DAW panel only, blurred along the movement of this frame
        if abs(jx[n]) + abs(jy[n]) > 0.5 or (n > 0 and abs(jx[n - 1]) + abs(jy[n - 1]) > 0.5):
            px, py = (jx[n - 1], jy[n - 1]) if n > 0 else (0.0, 0.0)
            if abs(jx[n]) + abs(jy[n]) > 0.5 and abs(px) + abs(py) < 0.5 and n in [fr(t) for t, _ in jolts]:
                px, py = 0.0, 0.0
            reg = np.ascontiguousarray(img[dy0:dy1])
            img[dy0:dy1] = motion_blur_path(reg, px, py, jx[n], jy[n])
        if s["layout"] == "face":
            img[FH - DIV:FH + DIV] = MINT
        # whips: outgoing frame streaks left, incoming frames streak in from the right
        tn = trans.get(n + 1)
        ti = trans.get(n)
        tp = trans.get(n - 1)
        if tn == "whipH":
            img = motion_blur_path(img, 0, 0, -160, 0, 14)
        elif tn == "whipV":
            img = motion_blur_path(img, 0, 0, 0, -220, 14)
        elif tn == "zoom":
            img = zoom_blur(img, 1.0, 1.35)
        if ti == "whipH":
            img = motion_blur_path(img, 220, 0, 80, 0, 14)
        elif ti == "whipV":
            img = motion_blur_path(img, 0, 300, 0, 100, 14)
        elif ti == "zoom":
            img = zoom_blur(img, 1.35, 1.12)
        if tp == "whipH":
            img = motion_blur_path(img, 80, 0, 0, 0, 10)
        elif tp == "whipV":
            img = motion_blur_path(img, 0, 100, 0, 0, 10)
        elif tp == "zoom":
            img = zoom_blur(img, 1.12, 1.0)
        if SHAKE and (abs(shx[n]) + abs(shy[n]) > 0.5):
            img = motion_blur_path(img, shx[n - 1] if n > 0 else 0.0, shy[n - 1] if n > 0 else 0.0, shx[n], shy[n], 10)
        if flash[n] > 0:
            img = cv2.addWeighted(img, 1 - flash[n], np.full_like(img, 255), flash[n], 0)
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        frames_done += 1
enc.stdin.close()
enc.wait()
json.dump({"start": START, "dur": DUR, "jolts": jolts, "whips": sorted(whip_in), "trans": {str(k): t for k, t in sorted(trans.items())},
           "shots": [{k_: s.get(k_) for k_ in ("t0", "t1", "layout", "view", "tau", "zm", "face_t", "cy", "n_cross")} for s in shots]},
          open(out_path.rsplit(".", 1)[0] + "_events.json", "w"), indent=1)
print("frames", frames_done, "shots", len(shots))
for s in shots:
    print(f"{s['t0']:6.2f} bar{START_BAR + s['j0']:5.2f} {s['layout']:5s} {s['view']:8s} zoom={s.get('zm')} cands={s['cands']}")
print("jolts", [(round(t, 2), a) for t, a in jolts], "whips", sorted(whip_in))
