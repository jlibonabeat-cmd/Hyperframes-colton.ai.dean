import subprocess, sys, json, numpy as np
exec(open("../scripts/cm_sync2.py").read().split("Cm,Fm=feats")[0])
Cm,Fm=feats(load("track.mp3"))
np.save("Cm.npy",Cm); np.save("Fm.npy",Fm)
out={}
for clip in sys.argv[1:]:
    Cc,Fc=feats(load(clip)); np.save(clip+".C.npy",Cc); np.save(clip+".F.npy",Fc)
    W=200; rows=[]
    for a in range(0,len(Cc)-W+1,50):
        L=W; sc=[]
        for j in range(0,len(Cm)-L):
            sc.append((Cc[a:a+L]*Cm[j:j+L]).sum(1).mean()+0.15*(Fc[a:a+L]*Fm[j:j+L]).mean())
        sc=np.array(sc); order=np.argsort(sc)[::-1]; top=[]
        for k in order:
            if all(abs(k-t)>30 for t in top): top.append(k)
            if len(top)==3: break
        rows.append((a*0.02,[(round(k*0.02,2),round(float(sc[k]),3)) for k in top]))
    print(clip)
    for a,t in rows: print(f"   clip {a:5.1f}s -> song start", t)
    out[clip]=rows
json.dump(out,open("windows.json","w"))
