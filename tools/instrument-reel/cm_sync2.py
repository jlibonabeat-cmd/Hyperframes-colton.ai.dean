"""Global offset per clip from chroma (harmony) + onset flux, full-clip and half-clip windows."""
import subprocess, sys, numpy as np
SR=22050; HOP=441  # 20 ms
def load(p):
    return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-vn","-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True).stdout,np.float32)
def feats(y):
    w=4096; n=(len(y)-w)//HOP
    F=np.fft.rfftfreq(w,1/SR); m=(F>70)&(F<2000); pc=(np.round(12*np.log2(F[m]/261.63))%12).astype(int)
    C=np.zeros((n,12)); fl=np.zeros(n); prev=None
    for i in range(n):
        X=np.abs(np.fft.rfft(y[i*HOP:i*HOP+w]*np.hanning(w)))
        C[i]=np.bincount(pc,weights=X[m],minlength=12)
        L=np.log1p(X[:800]*5)
        if prev is not None: fl[i]=np.maximum(0,L-prev).sum()
        prev=L
    C=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
    fl=fl-np.convolve(fl,np.ones(25)/25,mode="same"); fl=np.maximum(fl,0); fl=(fl-fl.mean())/(fl.std()+1e-9)
    return C,fl
Cm,Fm=feats(load("track.mp3"))
for clip in sys.argv[1:]:
    Cc,Fc=feats(load(clip))
    n=len(Cc)
    def score(a,b):
        seg=slice(a,b); L=b-a; out=[]
        for off in range(-a, len(Cm)-b):
            j=a+off
            ch=(Cc[seg]*Cm[j:j+L]).sum(1).mean()
            fx=(Fc[seg]*Fm[j:j+L]).mean()
            out.append((ch+0.15*fx, off))
        out.sort(reverse=True); return out
    res=score(0,n)
    best=res[0]; second=[r for r in res if abs(r[1]-best[1])>25][0]
    print(f"{clip}: full  D={best[1]*HOP/SR:7.2f}s score {best[0]:.3f}  next {second[0]:.3f} @ {second[1]*HOP/SR:.2f}")
    h=n//2
    for a,b in ((0,h),(h,n)):
        r=score(a,b); s2=[x for x in r if abs(x[1]-r[0][1])>25][0]
        print(f"    part {a*HOP/SR:5.1f}-{b*HOP/SR:5.1f}s D={r[0][1]*HOP/SR:7.2f} score {r[0][0]:.3f} next {s2[0]:.3f} @ {s2[1]*HOP/SR:.2f}")
