"""FEAR (110 BPM, B minor) - 9:16 J Sewell beat edit, song bars 9-20 (0:17.45-0:43.64): bars 9-16 the 808 + sample
chops loop (chops, jerk snare, new jazz 808 bass, clap, mixer), the drop at bar 17 (nocturne 808, hats, claps).
Green FL theme, OBS 1920x1200. Every view is a (recording, whole-bar shift) pair checked on the audio (fe_reuse.py ->
cover.json). Smooth-follow rolls (chops, snare, nocturne 808, bass): scroll speed + bar phase from the black bar lines,
playhead x chosen against the master onsets (fe_fx.py), drawn in FL's style where FL hides it. Page-mode rolls (hat,
clap): FL's dark-teal playhead. Picture sync: song(mp3) = clip + Dp(frame-exact FL clock) + 0.02.
Usage (from the fear folder): VARIANT=A|B python3 -I build_fe.py out.mp4
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
BPM = 110.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
B0 = 0.0                             # FL bar 1 = mp3 0:00 (offsets live in cover.json)
VARIANT = os.environ.get("VARIANT", "A")
START_J = 8                          # bar 9 (0:17.45)
NBARS = 12                           # bars 9-20 -> ends on the bar-21 downbeat (0:43.64)
START_BAR = START_J + 1
START = B0 + START_J * BAR
DUR = NBARS * BAR
NF = int(round(DUR * FPS))
FH = 800                             # face panel height (wide framing)
DIV = 3                              # half-height of the mint divider
MINT = (160, 230, 46)                # BGR
SRC_W, SRC_H = 1920, 1200
VARIANT = os.environ.get("VARIANT", "A")
# one face clip per edit (no outfit change mid-beat); only stretches where he's quietly listening at the screen
FACE_FILE, FACE_W, FACE_H, FACE_Y0 = "face_pac.mp4", 1080, 1920, 460
FACE_SEGS = [(0.6, 2.2), (3.4, 4.0), (5.0, 9.2), (9.9, 17.7), (18.6, 22.3), (22.9, 26.4)]   # same take as GLOCK
REACT_T = 19.0

out_path = sys.argv[1]
t_a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
t_b = float(sys.argv[3]) if len(sys.argv) > 3 else DUR
n_a, n_b = int(round(t_a * FPS)), min(NF, int(round(t_b * FPS)))


def fr(t):
    return int(round(t * FPS))


# ----------------------------------------------------------------------------- views (clip, instrument, kind, playback segments)
# segments: (clip_a, clip_b, D) with song = clip + D (picture), from kush_reuse.py (audio-checked per bar).
COVER = json.load(open(f"{HERE}/cover.json"))
ROLL = dict(kind="roll", lo=570, xmin=565, green=True, ymin=195, ymax=935, cy=560, h=600, hs=500, tz=1.0, fixcy=True)
FOLLOW = dict(kind="fixed", follow=True, ymin=195, ymax=935, tz=1.0, xmin=500, green=True, fixcy=True)   # fixcy: stray notes in other rows pulled the auto-centre off the chops
V = {}
for _vn, _segs in COVER.items():
    _base = _vn.split("_")[0]
    if not _segs:
        continue
    _sg = [tuple(x) for x in _segs]
    if _base in ("HAT", "CLAP"):                       # page mode: notes still, FL's playhead sweeps the page
        V[_vn] = dict(ROLL, clip={"HAT": "hat", "CLAP": "clap"}[_base] + ".mp4", inst="SNARE", segs=_sg)
        if _base == "HAT":
            V[_vn].update(cy=575, h=460, hs=420, pic=0.032)   # hat rows 534 / 615; page hits 32 ms late vs master hats
        if _base == "CLAP":
            V[_vn].update(cy=561, h=380, hs=360, pic=0.02)    # small clap ticks on one row: tight
    # Smooth-follow rolls: FL keeps the play position at the LEFT EDGE of the grid, hidden under the keyboard edge. The
    # piano roll's overview strip proves it (its play marker sits exactly on the left edge of the visible-window box),
    # and the black bar lines give the same x to a pixel. (v1 drew the line 1-2 bars further right: those notes looked
    # in time only where the pattern repeats bar to bar, the jerk snare was out of time. Jack: "totally out of time".)
    elif _base == "CHOPS":    # sample chops (slicer): keyboard is wider here, grid edge x=582 -> play line 583
        V[_vn] = dict(FOLLOW, clip="chops.mp4", inst="ARR", segs=_sg, ph=583, ph_lead=0.1, draw_ph=True, cy=790, h=470, hs=470)
    elif _base == "SNARE":    # jerk snare: grid edge 557 -> play line 559
        V[_vn] = dict(FOLLOW, clip="snare.mp4", inst="SNARE", segs=_sg, ph=559, ph_lead=0.1, draw_ph=True, cy=420, h=460, hs=440)
    elif _base == "N808":     # nocturne 808 (the drop): play line 558
        V[_vn] = dict(FOLLOW, clip="n808.mp4", inst="808", segs=_sg, ph=558, ph_lead=0.1, draw_ph=True, cy=561, h=520, hs=480)
    elif _base == "BASS":     # phantom new jazz II bass, smooth follow, FL's playhead visible at x=668
        V[_vn] = dict(FOLLOW, clip="bass.mp4", inst="808", segs=_sg, ph=668, cy=385, h=420, hs=400)
    elif _base == "MIXER":    # mixer: master + inserts 1-12 (melody, 808s, clap, snare) with names, meters, faders
        V[_vn] = dict(clip="mixer.mp4", inst="ARR", kind="fixed", segs=_sg, cy=480, h=700, fx=880, xmin=0)   # strips 510-1470: 808s, clap, snare, kick meters
    elif _base == "MACRO":    # arrangement, smooth follow with the playhead at x=849
        V["MACRO"] = dict(clip="macro.mp4", inst="ARR", kind="fixed", follow=True, segs=_sg, ph=849, ymin=140, ymax=1130,
                          cy=640, h=900, hs=900, tz=1.0, xmin=600)
_free_t = {}


PIC_LEAD = 0.0                       # offset fitted on the picture directly (note crossings vs master onsets)


import subprocess as _sp
from scipy.signal import stft as _stft
def _env(path, t0, dur, speed=1.0):
    y = np.frombuffer(_sp.run(["ffmpeg", "-v", "error", "-ss", f"{max(t0, 0):.3f}", "-t", f"{dur * speed:.3f}", "-i", path, "-ac", "1",
                               "-ar", "11025", "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
    f, _, Z = _stft(y, 11025, nperseg=512, noverlap=512 - 64)
    e = np.log1p(np.abs(Z)[(f >= 30) & (f < 160)].sum(0)); e = np.maximum(0, np.diff(e, prepend=e[0]))
    tt = np.arange(len(e)) * 64 / 11025 / speed
    return np.interp(np.arange(0, dur, 0.01), tt, e)
_warp_cache = {}
def coverage_warp(view, s0, s1):
    """clip time at s0 for a borrowed plugin view: KUSH song = (MERSEY bar position + n) * KUSH_BAR, n whole bars."""
    key = (view, round(s0, 3), round(s1, 3))
    if key in _warp_cache: return _warp_cache[key]
    v = V[view]; dur = s1 - s0
    em = _env(f"{HERE}/beat.mp3", s0, dur)
    best = None
    for a, b, D in v["runs"]:
        for n in range(-80, 120):
            sk0 = (s0 / BAR + n) * KUSH_BAR
            c0 = sk0 - D - 0.02; c1 = c0 + dur * WARP
            if c0 < a or c1 > b: continue
            ek = _env(f"{HERE}/{v['kbeat']}", sk0, dur, WARP)
            m = min(len(em), len(ek))
            sc = np.corrcoef(em[:m], ek[:m])[0, 1]
            if best is None or sc > best[0]: best = (sc, c0, n)
    _warp_cache[key] = None if best is None else best[1]
    if best: print(f"  {view} {s0:.2f}-{s1:.2f}: KUSH bar offset {best[2]}, low-end match {best[0]:.2f}")
    return _warp_cache[key]
def coverage(view, s0, s1):
    if V[view].get("warp"):
        return coverage_warp(view, s0, s1)
    if V[view].get("free"):
        a, b, _ = V[view]["segs"][0]
        if s1 - s0 > b - a: return None
        k = _free_t.setdefault((view, round(s0, 2)), len([1 for q in _free_t if q[0] == view]))
        return a + min(k * 1.3, b - a - (s1 - s0))
    for (a, b, D) in V[view]["segs"]:
        pl = V[view].get("pic", PIC_LEAD)
        if s0 - D - pl >= a - 0.08 and s1 - D - pl <= b + 0.06:   # +0.06: the 0.05 s look-ahead past a shot end   # <=80 ms before a run = playhead parked at its start
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
# the vox shot starts at beat 3.6: before that FL hasn't drawn its playhead yet (playback just starting).
# edit beats from bar 9 (0:09.70); drop = beat 4 (bar 10); section 2 = beat 36 (bar 18). Heard hits (edit beats):
# OK vox 3.7 4.7 20.2 20.7 | kick + bass 5.7 13.7 21.7 29.6 37.7 45.7 53.7 61.6 | 808 33.7 35.7 65.7 67.7 | perc 35.7 |
# rims 40.8 48.7 56.7 64.7 | snares: dense (follow view). Coverage (beats): MACRO 0-76 | VOX 3.6-37.3 | SNARE 5.7-27.8 |
# SNARE_8 37.7-59.8 | KICK 14.2-28.4, 30.2-35.3 | KICK_8 46.2-60.4, 62.2-67.3 | KICK_M2 6.2-20 | BASS 12.2-20.8, 22.8-28.7,
# 30.7-33.5 | BASS_8 44.2-52.8, 54.8-60.7, 62.7-65.5 | BASS_M2 4.2-8, 14.8-20 | BASS_2 20.2-28 | E808 59.1-71 |
# E808_M8 27.1-36 | PERC 35.7-37.7 | RIM 36-45.8, 48.1-51.4, 53.4-69.9.
# edit beats from bar 9 (0:17.45); the drop = beat 32 (bar 17). Coverage (edit beats): CHOPS 0.5-29.8 | SNARE 1.2-29.5 |
# BASS 8-13.5, 18-31.4 | CLAP 0.1-7.3, 12-13.7, 15.7-17.2, 32-42.7 | CLAP_4 16.1-20 | MIXER 3.1-39.6 | MACRO 0-38.5, 39.7-48
# | N808 32.1-42.6 | HAT 32-48. Starts on the chops (the hook), drop held on the nocturne 808 for ~3 s.
if VARIANT == "A":       # fast cut, face every other shot
    B_ = [0, 4, 8, 10, 12.5, 16.2, 20, 24, 27, 32, 37.5, 40.5, 44, 48]
    L_ = ["face:CHOPS_m4", "full:SNARE", "face:MIXER", "full:BASS", "face:CHOPS", "full:CLAP_4", "face:SNARE", "full:BASS",
          "face:MIXER", "full:N808", "face:CLAP", "full:MACRO", "face:HAT"]
    REACT_J = 37.5 / 4
else:                    # longer holds, wide arrangement first
    B_ = [0, 2.5, 8, 12, 18, 22, 26, 29.5, 32, 38, 41, 48]
    L_ = ["full:MACRO", "face:CHOPS", "full:BASS", "face:SNARE", "full:MIXER", "face:BASS", "full:CHOPS", "face:MIXER",
          "full:N808", "full:CLAP", "face:HAT"]
    REACT_J = 38 / 4
CUT_BARS = [b / 4 for b in B_[:-1]] + [NBARS]
FORCE = {b / 4: tuple(l.split(":")) for b, l in zip(B_, L_)}
DROP_J = 8                                   # bar 17: the drop
# the real impact: after "oh" (beat 3.7) / "okay" (4.7) the beat drops out (beats 5.05-5.65, -29 dB) and slams in at 5.66
# (kick + bass + 808). C: J Sewell blackout from the "oh" through "okay" and the gap, back on the hit with the flash
# (Jack: "when he goes oh it needs to go black and stay black until ok has finished, then the beat drops").
DROP_T = DROP_J * BAR
# J Sewell blackout (Jack loved it on MERSEY): after the counter snare at the end of bar 9 the beat drops out (10.76-11.19 s,
# -34 dB, the OK vox inside it) and slams back at bar 10 -> black for exactly that gap.
BLACK = None
TRANS_FORCE = {}
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
VAL = json.load(open(f"{HERE}/validate_x.json"))    # sd_reuse.py: picture hits shifted into the edit
for x in shots:
    vv = VAL.get(x["view"])
    if not vv:
        continue
    s0_, s1_ = START + x["t0"], START + x["t1"]
    ys = [c[1] for c in vv["cross"] if s0_ <= c[0] < s1_]
    x["n_cross"] = len(ys)
    if ys:
        if not V[x["view"]].get("fixcy"):
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
    else:                                     # clean footage used up: reuse a clean stretch (second pass, offset)
        fq2 = globals().setdefault("_fq2", [[g[0] + 0.7, g[1]] for g in FACE_SEGS])
        fits = [seg for seg in fq2 if seg[1] - seg[0] >= need]
        if not fits:
            raise SystemExit(f"not enough clean face footage for shot at {s_['t0']:.2f}")
        seg = max(fits, key=lambda g: g[1] - g[0])
        s_["face_t"] = seg[0]; s_["face_reuse"] = True
        seg[0] += need

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
    elif s["view"].split("_")[0] in ("MIXER", "PLUGIN", "CLIP", "COMP", "CLIPK", "PLUGK") or s["view"].startswith("MIXER_T"):
        s["zm"] = "mixer"
    elif s["layout"] == "full" and (v["kind"] == "roll" or v.get("follow")):
        s["zm"] = "track"                       # tight macro, camera follows the playhead
    elif s["layout"] == "full" and s["view"] == "MACRO_W":
        s["zm"] = "push"
    elif s["layout"] == "full" and s["view"] in ("MACRO", "MACRO_T", "MACRO_D", "MACRO_P", "MACRO_G", "MACRO_DR", "MACRO_KIT", "MACRO_K", "FUNK", "PREDROP") and s["j0"] not in (DROP_J,):
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
drop_t = DROP_T
cand = [(s, t) for t, s in zip(kt, ks) if t > drop_t + 0.5 and t < DUR - 2 * BAR and s >= 0.45]
cand.sort(reverse=True)
jolts = []   # the drop itself gets a whip + soft flash, not a jolt
for s, t in cand:
    sh_ = [x for x in shots if x["t0"] <= t < x["t1"]]
    near_hit = (not sh_) or any(abs(t - h) < 0.25 for h in view_hits(sh_[0]["view"]))
    if not near_hit and all(abs(t - u) > 2.0 for u, _ in jolts) and all(abs(fr(t) - w) > 6 for w in whip_in) and len(jolts) < 2:
        jolts.append((float(t), 0.8))
jolts.sort()
jolts = []                                   # Jack: no shake on the 808s
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
        if kind_ in ("hard", "soft"):
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


def reader(path, ss, nframes, w, h_, speed=1.0):
    vf = f"fps={FPS}" if speed == 1.0 else f"setpts=(PTS-STARTPTS)/{speed:.5f},fps={FPS}"
    cmd = ["ffmpeg", "-nostats", "-v", "error", "-ss", f"{max(ss, 0):.3f}", "-i", path, "-vf", vf,
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


def find_playhead(frm, kind, lo=None, light=False, rows=None, pht=0.75, beige=False, purple=False, pink=False, brown=False, green=False):
    if kind == "fixed":
        return None
    if green:                                # green FL theme: thin dark-teal playhead (~R9 G43 B46)
        reg = frm[200:930:3].astype(np.int16)
        B_, G, R = reg[..., 0], reg[..., 1], reg[..., 2]
        g = (R < 28) & (G > 36) & (B_ > 36) & (G < 90) & (B_ < 90) & (np.abs(B_ - G) < 20)
        c = np.where(g.mean(axis=0)[lo:1888] > 0.5)[0]
        return float(np.median(c)) + lo if len(c) and c.max() - c.min() < 12 else None
    if brown:                                # brown FL theme: light beige 3-px playhead with an orange glow
        reg = frm[205:890:3].astype(np.int16)
        R, G, B_ = reg[..., 2], reg[..., 1], reg[..., 0]
        g = (R > 140) & (G > 118) & (R - G < 35) & (R - B_ > 20)
        c = np.where(g.mean(axis=0)[lo:1885] > 0.55)[0]
        return float(np.median(c)) + lo if len(c) else None
    if pink:                                 # pink FL theme: playhead core and notes are hot pink; the playhead is the
        reg = (frm[rows[0]:rows[1]:2] if rows else frm[190:885:3]).astype(np.int16)  # only column covering most grid rows
        R, G, B_ = reg[..., 2], reg[..., 1], reg[..., 0]
        g = (R > 175) & (R - G > 75) & (B_ > 110) & (B_ < 220)
        c = np.where(g.mean(axis=0)[lo:1885] > 0.55)[0]
        return int(c[len(c) // 2]) + lo if len(c) else None
    if purple:                               # purple FL theme: playhead and notes share a pink-purple; the playhead
        reg = frm[190:885:3].astype(np.int16)  # is the only column covering most grid rows
        g = (reg[..., 0] > 190) & (reg[..., 2] > 150) & (reg[..., 1] > 55) & (reg[..., 1] < 130)
        c = np.where(g.mean(axis=0)[lo:1885] > 0.55)[0]
        return int(c[len(c) // 2]) + lo if len(c) else None
    if beige:                                # beige FL theme: red playhead
        reg = frm[260:940:4].astype(np.int16)
        g = (reg[..., 2] - reg[..., 1] > 45) & (reg[..., 2] > 200)
        c = np.where(g.mean(axis=0)[lo:1890] > 0.6)[0]
        return int(c[len(c) // 2]) + lo if len(c) else None
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
            x0 = v["ph"] - v.get("ph_lead", 0.35) * cw
        elif "fxl" in v:                   # piano roll in smooth-follow: notes stream left and play at the keyboard edge
            x0 = v["fxl"]
        else:
            x0 = v["fx"] - cw / 2
    else:
        x0 = x - lead * cw
    x0 = min(max(x0, xmin), 1888 - cw)
    y0 = min(max(v["cy"] - ch / 2, v.get("ymin", 100)), v.get("ymax", 1040) - ch) if ch < 940 else 100
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
        elif v.get("green"):                 # green theme: teal glow, teal core (FL's playhead, a touch brighter)
            img[y0:y1, a:X] = (reg * 0.55 + np.array((70, 64, 12), np.float32) * 0.45).astype(np.uint8)
            img[y0:y1, X:min(size[0], X + max(3, int(1.2 * sx)))] = (150, 140, 40)
        else:                                # brown theme: orange glow, light beige core (as FL draws it)
            img[y0:y1, a:X] = (reg * 0.45 + np.array((36, 70, 130), np.float32) * 0.55).astype(np.uint8)
            img[y0:y1, X:min(size[0], X + max(3, int(1.2 * sx)))] = (126, 149, 170)


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
    rd = reader(f"{HERE}/{v['clip']}", s["tau"] + off * (WARP if v.get("warp") else 1.0), k, SRC_W, SRC_H, WARP if v.get("warp") else 1.0)
    frd = reader(f"{HERE}/{FACE_FILE}", s["face_t"] + off, k, FACE_W, FACE_H) if s["layout"] == "face" else None
    st = {}
    lastx = [None]
    for n in range(f0, f1):
        src = next(rd)
        x = find_playhead(src, v["kind"], v.get("lo"), v.get("light", False), v.get("ph_rows"), v.get("pht", 0.75), v.get("beige", False), v.get("purple", False), v.get("pink", False), v.get("brown", False), v.get("green", False))
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
        if (s.get("zm") == "track" or (s["layout"] == "face" and v["kind"] == "roll")) and x is not None:
            xs_ = st.get("xs")
            xs_ = x if xs_ is None else 0.3 * xs_ + 0.7 * x      # light smoothing: the zoomed hat roll moves ~35 px/frame in page mode
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
            if v["kind"] == "roll" and st.get("xs") is not None:
                r = view_rect(v, st["xs"], W / bh, lead=0.45)    # split panel rides the playhead too (no page jumps)
            else:
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
        elif tn == "soft":
            img = motion_blur_path(img, 0, 0, -50, 0, 8)
        if ti == "whipH":
            img = motion_blur_path(img, 220, 0, 80, 0, 14)
        elif ti == "whipV":
            img = motion_blur_path(img, 0, 300, 0, 100, 14)
        elif ti == "zoom":
            img = zoom_blur(img, 1.35, 1.12)
        elif ti == "soft":
            img = motion_blur_path(img, 70, 0, 25, 0, 8)
        if tp == "whipH":
            img = motion_blur_path(img, 80, 0, 0, 0, 10)
        elif tp == "whipV":
            img = motion_blur_path(img, 0, 100, 0, 0, 10)
        elif tp == "zoom":
            img = zoom_blur(img, 1.12, 1.0)
        elif tp == "soft":
            img = motion_blur_path(img, 25, 0, 0, 0, 6)
        if SHAKE and (abs(shx[n]) + abs(shy[n]) > 0.5):
            img = motion_blur_path(img, shx[n - 1] if n > 0 else 0.0, shy[n - 1] if n > 0 else 0.0, shx[n], shy[n], 10)
        if BLACK and any(fr(a_) <= n < fr(b_) for a_, b_ in BLACK):
            img = np.zeros_like(img)                 # blackout: the beat drops out
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
