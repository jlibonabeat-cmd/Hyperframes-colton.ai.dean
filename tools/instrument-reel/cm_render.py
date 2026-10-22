"""Hard-cut instrument reel: each shot = proxy clip from c0 for the shot's exact frame count; master audio only."""
import json, subprocess, sys
FPS=30
plan=json.load(open("plan.json")); out=sys.argv[1]
END=plan[-1]["song1"]; NF=round(END*FPS)
parts=[]; n_prev=0
for i,s in enumerate(plan):
    n1=round(s["song1"]*FPS) if i<len(plan)-1 else NF
    nfr=n1-n_prev; n_prev=n1
    seg=f"seg_{i:02d}.mp4"
    subprocess.run(["ffmpeg","-y","-v","error","-ss",f"{s['c0']:.3f}","-i",s["clip"],"-frames:v",str(nfr),"-an",
                    "-vf","fps=30,setsar=1","-c:v","libx264","-crf","16","-preset","medium","-pix_fmt","yuv420p",seg],check=True)
    parts.append(seg)
open("concat.txt","w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg","-y","-v","error","-f","concat","-safe","0","-i","concat.txt","-ss","0","-t",f"{NF/FPS+0.05:.3f}","-i","track.mp3",
                "-map","0:v","-map","1:a","-c:v","libx264","-crf","17","-preset","medium","-pix_fmt","yuv420p","-r","30",
                "-af",f"afade=t=out:st={NF/FPS-0.08:.3f}:d=0.08","-c:a","aac","-b:a","256k","-movflags","+faststart","-shortest",out],check=True)
print("frames",NF,"dur",NF/FPS)
