"""Hit list for the edit window with strengths. Writes hits.json (times relative to window start)."""
import json, subprocess, sys
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
SR=44100; START=float(sys.argv[1]); DUR=float(sys.argv[2])
raw=subprocess.run(["ffmpeg","-nostats","-v","error","-ss",str(START),"-t",str(DUR),"-i","beat.mp3","-vn","-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True,check=True).stdout
y=np.frombuffer(raw,dtype=np.float32)
def band(lo,hi): return sosfilt(butter(4,[lo,hi],btype="band",fs=SR,output="sos"),y)
def curve(x,hop=441,win=441):
    e=np.convolve(np.abs(x),np.ones(win)/win,mode="same")[::hop]
    le=np.log1p(e*200); return np.maximum(0,np.diff(le,prepend=le[0]))
def peaks(c,gap,pct):
    thr=np.percentile(c,pct); pk,_=find_peaks(c,height=thr,distance=int(gap*100))
    return pk/100.0, c[pk]/c[pk].max()
out={}
for name,(lo,hi,gap,pct) in {"kick":(35,130,0.10,90),"snare":(1500,5000,0.10,92),"hat":(7000,16000,0.05,85)}.items():
    t,s=peaks(curve(band(lo,hi)),gap,pct); out[name]={"t":[round(float(a),3) for a in t],"s":[round(float(a),3) for a in s]}
json.dump(out,open("hits.json","w"))
for k,v in out.items(): print(k,len(v["t"]))
