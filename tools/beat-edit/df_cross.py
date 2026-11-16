"""DRUG FIT sync check from the picture: times where a note's left edge crosses FL's play position (marker x) ->
song time (clock) vs master onsets in the band. Tracks notes frame to frame by row + nearest x (handles scrolling).
usage: df_cross.py clip band [ph_fixed]"""
import json, sys, numpy as np
clip, bk = sys.argv[1], sys.argv[2]
S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]; on = np.array(json.load(open("ons_master.json"))[{"snr": "snare", "sub": "kick", "hi": "hat"}[bk]]["t"])
cr = []
for i in range(1, len(S)):
    p0, p1 = S[i - 1][0], S[i][0]
    if p0 is None or p1 is None or abs(p1 - p0) > 40: continue
    a = [(b[0], b[2]) for b in S[i - 1][1]]; c = [(b[0], b[2]) for b in S[i][1]]
    for x1, y1 in c:
        cand = [(abs(x0 - x1), x0) for x0, y0 in a if abs(y0 - y1) <= 3 and abs(x0 - x1) < 30]
        if not cand: continue
        x0 = min(cand)[1]
        d0, d1 = x0 - p0, x1 - p1
        if d0 > 0 >= d1 and d0 - d1 > 0:
            t = (i - 1 + d0 / (d0 - d1)) / 60
            D = next((r[2] for r in C if r[0] <= t <= r[1]), None)
            if D is not None: cr.append((t + D + 0.02, y1))
cr = sorted(set((round(t, 4), y) for t, y in cr))
d = np.array([(t - on[np.argmin(abs(on - t))]) * 1000 for t, _ in cr]) if cr else np.array([np.nan])
print(f"{clip}: {len(cr)} crossings; vs master {bk}: median {np.nanmedian(d):+.1f} ms, <25 ms {(abs(d) < 25).mean():.2f}, <40 ms {(abs(d) < 40).mean():.2f}; worst {sorted(d, key=abs)[-3:] if len(d) else ''}")
json.dump(cr, open(f"{clip}_cross.json", "w"))
