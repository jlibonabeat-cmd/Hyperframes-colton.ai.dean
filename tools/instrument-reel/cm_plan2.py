import json, numpy as np, sys
sys.argv=["x"]
exec(open("../scripts/cm_plan.py").read().split('if __name__=="__main__":')[0])
HB=BAR/2
take_D={"W":9.62,"H":8.205,"S":7.98,"F":12.79}   # each take's offset (song = clip + D, mod 4 bars), from whole-clip matching                                   # chords change every half bar (1.333 s at 90 BPM)
CUT=[0,1,2,4,6,7,8,10,12,14,16]            # in half bars -> song 0..21.33 (the 8-bar riff section)
ORDER=["W","H","S","F","H","W","S","H","F","W"]
used={k:[] for k in CLIPS}
plan=[]
for i in range(len(CUT)-1):
    s0,s1=CUT[i]*HB,CUT[i+1]*HB
    pref=ORDER[i]
    opts=[]
    for k in CLIPS:
        if plan and plan[-1]["angle"]==k: continue
        c0,sc=best_pos(k,s0,s1,used[k]); opts.append((sc+(0.3 if k==pref else 0),k,c0,sc))
    opts.sort(reverse=True); _,k,c0,sc=opts[0]
    used[k].append((c0,c0+(s1-s0)))
    plan.append(dict(song0=round(s0,4),song1=round(s1,4),angle=k,clip=CLIPS[k],c0=round(c0,3),score=round(sc,3)))
    print(f"song {s0:5.2f}-{s1:5.2f}  {k} {CLIPS[k]} @ {c0:6.3f}  score {sc:.3f}   (pref {pref}; others {[(o[1],round(o[3],2)) for o in opts[1:]]})")
json.dump(plan,open("plan.json","w"),indent=1)
