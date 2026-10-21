import json, subprocess, numpy as np
def find_playhead(frm, kind):
    reg = frm[200:880:4].astype(np.int16)
    g = (reg[..., 1] - reg[..., 0] > 35) & (reg[..., 1] >= reg[..., 2] - 5)
    col = g.mean(axis=0)
    lo = 500 if kind == "roll" else 760
    c = np.where(col[lo:1890] > 0.75)[0]
    return int(c[0]) + lo if len(c) else None
out = {}
for clip, kind in [("hat", "roll"), ("openhat", "roll"), ("snare_a", "roll"), ("snare_main", "roll"), ("s808", "roll"), ("s808b", "roll"), ("macro", "play")]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{clip}.mp4", "-vf", "fps=10", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    xs = []
    while True:
        b = p.stdout.read(1080 * 1920 * 3)
        if len(b) < 1080 * 1920 * 3:
            break
        xs.append(find_playhead(np.frombuffer(b, np.uint8).reshape(1080, 1920, 3), kind))
    p.wait()
    out[clip] = xs
    pres = "".join("|" if x is not None else "." for x in xs)
    print(clip, pres)
json.dump(out, open("playhead.json", "w"))
