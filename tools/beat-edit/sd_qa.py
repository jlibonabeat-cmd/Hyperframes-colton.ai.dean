"""QA for the SO DOPE renders: per-shot sheet (4 frames per shot), empty-roll scan (<6 pink px in the roll panel for
>=12 frames), frozen-picture scan (mean diff <0.15 for >=12 frames), dark-edge count. usage: sd_qa.py A B"""
import cv2, numpy as np, json, sys
for v in sys.argv[1:]:
    d = json.load(open(f'out/sd_{v}_events.json')); c = cv2.VideoCapture(f'out/sd_{v}.mp4'); shots = []; sh = {}
    for s in d['shots']:
        for n in range(int(round(s['t0'] * 30)), int(round(s['t1'] * 30))): sh[n] = s
        for u in (0.08, 0.35, 0.65, 0.92):
            n = int((s['t0'] + (s['t1'] - s['t0']) * u) * 30); c.set(1, n); ok, f = c.read(); f = cv2.resize(f, (108, 192))
            cv2.putText(f, f"{n/30:.1f} {s['view'][:7]}", (2, 12), 0, 0.33, (0, 0, 255), 1); shots.append(f)
    rows = [np.hstack(shots[i:i + 16] + [np.zeros_like(shots[0])] * (16 - len(shots[i:i + 16]))) for i in range(0, len(shots), 16)]
    cv2.imwrite(f'sheets/{v}_shots.png', np.vstack(rows))
    cap = cv2.VideoCapture(f'out/sd_{v}.mp4'); prev = None; fz = []; i = 0; empty = []; edge = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        s = sh.get(i); mm = f.mean(2); edge += sum(sl.mean() < 8 for sl in (mm[:6], mm[-6:], mm[:, :6], mm[:, -6:]))
        if s and not s['view'].startswith('MACRO'):
            p = (f[800:] if s['layout'] == 'face' else f)[::4, ::4].astype(int); R, G, B = p[..., 2], p[..., 1], p[..., 0]
            m = (R > 175) & (R - G > 75) & (B > 110) & (B < 220); colc = m.sum(0); m[:, colc > m.shape[0] * 0.4] = False
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
    print(v, i, 'shots', len(d['shots']), 'macro', sum(s['view'].startswith('MACRO') for s in d['shots']), 'edges', edge,
          'empty', runs(empty, 12), 'frozen', runs(fz, 12))
