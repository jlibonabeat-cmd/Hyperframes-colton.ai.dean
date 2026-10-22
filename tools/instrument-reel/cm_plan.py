"""For each shot (song window), find the clip + clip time whose own audio best matches the master (onset flux + chroma)."""
import subprocess, json, numpy as np
SR=22050; HOP=110; DT=HOP/SR   # ~5 ms
def load(p):
    return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-vn","-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True).stdout,np.float32)
def feats(y):
    w=2048; n=(len(y)-w)//HOP
    idx=np.arange(w)[None,:]+HOP*np.arange(n)[:,None]
    X=np.abs(np.fft.rfft(y[idx]*np.hanning(w),axis=1))
    L=np.log1p(X[:,5:400]*10)
    f=np.maximum(0,np.diff(L,axis=0)).sum(1); f=np.concatenate([[0],f])
    f=f-np.convolve(f,np.ones(40)/40,mode="same"); f=np.maximum(f,0)
    F=np.fft.rfftfreq(w,1/SR); m=(F>70)&(F<1800); pc=(np.round(12*np.log2(F[m]/261.63))%12).astype(int)
    C=np.stack([X[:,m][:,pc==k].sum(1) for k in range(12)],1)
    C=np.convolve  # placeholder
    return f, X[:,m], pc
def chroma(Xm,pc):
    C=np.stack([Xm[:,pc==k].sum(1) for k in range(12)],1)
    C=np.array([np.convolve(C[:,k],np.ones(20)/20,mode="same") for k in range(12)]).T
    return C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
fm,Xm,pc=feats(load("track.mp3")); Cm=chroma(Xm,pc)
CLIPS={"W":"IMG_9441.mp4","H":"IMG_9446.mp4","S":"IMG_9447.mp4","F":"IMG_9451.mp4"}
data={}
for k,p in CLIPS.items():
    f,X,_=feats(load(p)); data[k]=(f,chroma(X,pc))
BAR=4*60/90
take_D=None
def best_pos(k,s0,s1,avoid=()):
    f,C=data[k]; a=int(round(s0/DT)); L=int(round((s1-s0)/DT))
    tgt=fm[a:a+L]; tc=Cm[a:a+L]
    tz=(tgt-tgt.mean())/(tgt.std()+1e-9)
    sc=[]
    for c0 in range(0,len(f)-L):
        seg=f[c0:c0+L]; z=(seg-seg.mean())/(seg.std()+1e-9)
        r=(z*tz).mean(); ch=(C[c0:c0+L]*tc).sum(1).mean()
        sc.append(r+0.5*ch)
    sc=np.array(sc)
    if take_D is not None:                  # stay on this take's beat grid: song - clip == D (mod 4 bars) +- tol
        P=4*BAR
        cs=np.arange(len(sc))*DT
        d=np.mod(s0-cs-take_D[k]+P/2,P)-P/2
        sc[np.abs(d)>0.12]=-9
    for (u0,u1) in avoid:                   # don't reuse the same moment of a clip
        sc[max(0,int(u0/DT)-L):int(u1/DT)]=-9
    k_=int(np.argmax(sc)); return k_*DT, float(sc[k_])
if __name__=="__main__":
    import sys
    CUT=[0,1,2,4,6,8,10,12,14,16]
    for i in range(len(CUT)-1):
        s0,s1=CUT[i]*BAR,CUT[i+1]*BAR
        row=[]
        for k in CLIPS:
            c0,sc=best_pos(k,s0,s1); row.append(f"{k}:{c0:6.2f}({sc:.2f})")
        print(f"bars {CUT[i]:2d}-{CUT[i+1]:2d} song {s0:5.2f}-{s1:5.2f} ", "  ".join(row))
