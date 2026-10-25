"""Windowed chroma+onset match of each clip's audio against the master (whole track). Prints top song positions per 3 s window."""
import subprocess, sys, numpy as np
SR=22050; HOP=441
def load(p):
    return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",p,"-vn","-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True).stdout,np.float32)
def feats(y):
    w=4096; n=(len(y)-w)//HOP
    idx=np.arange(w)[None,:]+HOP*np.arange(n)[:,None]
    X=np.abs(np.fft.rfft(y[idx]*np.hanning(w),axis=1))
    F=np.fft.rfftfreq(w,1/SR); m=(F>70)&(F<2000); pc=(np.round(12*np.log2(F[m]/261.63))%12).astype(int)
    C=np.stack([X[:,m][:,pc==k].sum(1) for k in range(12)],1); C/= (np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
    L=np.log1p(X[:,:800]*5); fl=np.concatenate([[0],np.maximum(0,np.diff(L,axis=0)).sum(1)])
    fl=fl-np.convolve(fl,np.ones(25)/25,mode="same"); fl=np.maximum(fl,0); fl=(fl-fl.mean())/(fl.std()+1e-9)
    return C,fl
Cm,Fm=feats(load("track.mp3"))
for clip in sys.argv[1:]:
    Cc,Fc=feats(load(clip)); W=150
    print(clip)
    for a in range(0,len(Cc)-W+1,50):
        sc=np.array([(Cc[a:a+W]*Cm[j:j+W]).sum(1).mean()+0.15*(Fc[a:a+W]*Fm[j:j+W]).mean() for j in range(0,len(Cm)-W)])
        order=np.argsort(sc)[::-1]; top=[]
        for k in order:
            if all(abs(k-t)>25 for t in top): top.append(k)
            if len(top)==3: break
        print(f"   clip {a*0.02:5.1f}s ->", [(round(k*0.02,2),round(float(sc[k]),3), "D=%.2f"%(k*0.02-a*0.02)) for k in top])
