"""DRUG FIT note-edit finder for page-mode rolls: a note box appearing / disappearing / resizing away from the play
position (>30 px), between frames 6 apart, with no page flip in between. usage: df_edits.py clip..."""
import json, sys
for clip in sys.argv[1:]:
    S = json.load(open(f"{clip}_scan.json")); C = json.load(open("clock2.json"))[clip]["runs"]; ev = []
    for i in range(6, len(S), 3):
        p0, p1 = S[i - 6][0], S[i][0]
        if p0 is not None and p1 is not None and abs(p1 - p0) > 60: continue          # page flip / jump
        a = {(b[0] // 3, b[1] // 3, b[2] // 3) for b in S[i - 6][1]}; c = {(b[0] // 3, b[1] // 3, b[2] // 3) for b in S[i][1]}
        def near(k): return any(abs(k[0] * 3 - p) < 30 or abs(k[1] * 3 - p) < 30 for p in (p0, p1) if p is not None)
        def fuzzy_in(k, s): return any(abs(k[0] - q[0]) <= 1 and abs(k[1] - q[1]) <= 1 and abs(k[2] - q[2]) <= 1 for q in s)
        add = [k for k in c if not fuzzy_in(k, a) and not near(k)]; rem = [k for k in a if not fuzzy_in(k, c) and not near(k)]
        if add or rem: ev.append((round(i / 60, 2), [(k[0] * 3, k[2] * 3) for k in add], [(k[0] * 3, k[2] * 3) for k in rem]))
    m = []
    for t, ad, rm in ev:
        if m and t - m[-1][1] < 0.35: m[-1][1] = t; m[-1][2] += ad; m[-1][3] += rm
        else: m.append([t, t, list(ad), list(rm)])
    runs = [(round(r[0], 2), round(r[1], 2)) for r in C]
    print(clip, "runs", runs, "\n   edits:", [(a, b, ad[:2], rm[:2]) for a, b, ad, rm in m])
