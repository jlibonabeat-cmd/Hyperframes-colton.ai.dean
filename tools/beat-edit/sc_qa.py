"""QA for the SCOUSE renders: per-shot sheet (5 frames per shot, 2 shots a row), punch sheet (frame before, on and
2 after each hit punch), empty-roll scan (<6 mint note px in the roll panel for >=12 frames), frozen-picture scan,
dark-edge count. usage: sc_qa.py A B"""
import cv2, numpy as np, json, sys
TW, TH = 144, 256
def grab(c, n, lab):
    c.set(1, n); ok, f = c.read(); f = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
    cv2.putText(f, lab, (2, 12), 0, 0.36, (0, 0, 255), 1); return f
for v in sys.argv[1:]:
    d = json.load(open(f'out/scouse_{v}_events.json')); c = cv2.VideoCapture(f'out/scouse_{v}.mp4'); th = []; sh = {}
    for s in d['shots']:
        for n in range(int(round(s['t0'] * 30)), int(round(s['t1'] * 30))): sh[n] = s
        for u in (0.04, 0.27, 0.5, 0.73, 0.96):
            n = int((s['t0'] + (s['t1'] - s['t0']) * u) * 30); th.append(grab(c, n, f"{n/30:.2f} {s['view'][:8]}"))
    rows = [np.hstack(th[i:i + 10] + [np.zeros_like(th[0])] * (10 - len(th[i:i + 10]))) for i in range(0, len(th), 10)]
    cv2.imwrite(f'sheets/{v}_shots.png', np.vstack(rows))
    pt = []
    for t, a in d['punches'] + d.get('impacts', []):
        n = int(round(t * 30)); pt += [grab(c, m, f"{m/30:.2f}{'*' if m == n else ''}") for m in (n - 1, n, n + 1, n + 2)]
    if pt:
        rows = [np.hstack(pt[i:i + 12] + [np.zeros_like(pt[0])] * (12 - len(pt[i:i + 12]))) for i in range(0, len(pt), 12)]
        cv2.imwrite(f'sheets/{v}_punch.png', np.vstack(rows))
    cap = cv2.VideoCapture(f'out/scouse_{v}.mp4'); prev = None; fz = []; i = 0; empty = []; edge = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        s = sh.get(i); mm = f.mean(2); edge += sum(sl.mean() < 8 for sl in (mm[:6], mm[-6:], mm[:, :6], mm[:, -6:]))
        if s and s['view'].split('_')[0] in ('F808', 'M808', 'HAT', 'KICK', 'CLAP', 'CLAP4', 'OHAT', 'S808'):
            p = (f[800:] if s['layout'] == 'face' else f)[::4, ::4].astype(int); R, G, B = p[..., 2], p[..., 1], p[..., 0]
            m = (G > 170) & (R > 115) & (B > 115) & (G - R > 15) & (G - B > 8)
            if m.sum() < 6: empty.append(i)
        g = cv2.resize(f, (108, 192)).astype(int)
        if prev is not None and abs(g - prev).mean() < 0.15: fz.append(i)
        prev = g; i += 1
    def runs(xs, mn):
        r = []
        for x in xs:
            if r and x - r[-1][1] <= 1: r[-1][1] = x
            else: r.append([x, x])
        return [(round(a / 30, 2), round(b / 30, 2), sh.get(a, {}).get('view')) for a, b in r if b - a >= mn]
    print(v, i, 'frames, shots', len(d['shots']), 'edges', edge, 'empty', runs(empty, 12), 'frozen', runs(fz, 12))
