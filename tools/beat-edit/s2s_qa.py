"""QA for the SOMETHING2SAY renders: shot sheet (5 frames per shot, 2 shots a row), empty-roll scan (pale-green note
pixels in the roll panel < 40 for >= 12 frames), frozen-picture scan (>= 12 near-identical frames in the DAW panel),
dark-edge count. usage (from the s2s folder): python3 -I scripts/s2s_qa.py A B"""
import cv2, numpy as np, json, sys
TW, TH = 216, 384
def grab(c, n, lab):
    c.set(1, n); ok, f = c.read(); f = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
    cv2.putText(f, lab, (3, 14), 0, 0.45, (0, 0, 255), 1); return f
ROLLS = ("BASS", "CHORDS", "COUNTER", "SERUM")
for v in sys.argv[1:]:
    d = json.load(open(f'out/S2S_{v}_events.json')); c = cv2.VideoCapture(f'out/S2S_{v}.mp4'); th = []; sh = {}
    for s in d['shots']:
        for n in range(int(round(s['t0'] * 30)), int(round(s['t1'] * 30))): sh[n] = s
        for u in (0.03, 0.27, 0.5, 0.73, 0.97):
            n = int((s['t0'] + (s['t1'] - s['t0']) * u) * 30); th.append(grab(c, n, f"{n/30:.2f} {s['view'][:9]}"))
    rows = [np.hstack(th[i:i + 10] + [np.zeros_like(th[0])] * (10 - len(th[i:i + 10]))) for i in range(0, len(th), 10)]
    for k in range(0, len(rows), 3):
        cv2.imwrite(f'sheets/{v}_shots{k // 3}.png', np.vstack(rows[k:k + 3]))
    cap = cv2.VideoCapture(f'out/S2S_{v}.mp4'); prev = None; fz = []; i = 0; empty = []; edge = 0; npx = []
    while True:
        ok, f = cap.read()
        if not ok: break
        s = sh.get(i); mm = f.mean(2); edge += sum(sl.mean() < 8 for sl in (mm[:6], mm[-6:], mm[:, :6], mm[:, -6:]))
        p = (f[806:] if s and s['layout'] == 'face' else f)
        if s and s['view'].split('_')[0] in ROLLS:
            q = p[::3, ::3].astype(int); R, G, B = q[..., 2], q[..., 1], q[..., 0]
            m = (G > 180) & (R > 120) & (G - R > 15) & (G - B > 5); npx.append((i, int(m.sum())))
            empty.append(i if m.sum() < 40 else None)
        else:
            empty.append(None)
        g = cv2.cvtColor(cv2.resize(p, (135, 140)), cv2.COLOR_BGR2GRAY).astype(int)
        fz.append(prev is not None and np.abs(g - prev).mean() < 0.6); prev = g; i += 1
    def runs(flags, mn=12):
        out, st = [], None
        for k, x in enumerate(list(flags) + [False]):
            if x and st is None: st = k
            if not x and st is not None:
                if k - st >= mn: out.append((round(st / 30, 2), round(k / 30, 2)))
                st = None
        return out
    print(v, 'frames', i, '| empty-roll runs', runs([e is not None for e in empty]), '| frozen runs', runs(fz),
          '| dark edges', edge, '| min note px per roll shot',
          {s['view'] + '@' + str(round(s['t0'], 1)): min([x[1] for x in npx if sh.get(x[0]) is s] or [-1]) for s in d['shots'] if s['view'].split('_')[0] in ROLLS})
