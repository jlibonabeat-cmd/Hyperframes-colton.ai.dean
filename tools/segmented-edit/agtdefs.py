"""Load build_agt.py's views + coverage() without rendering (exec up to the audio-features block)."""
import sys, os
src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "build_agt.py")).read()
src = src[:src.index("# ----------------------------------------------------------------------------- audio features")]
_argv = sys.argv; sys.argv = ["x", "/tmp/_dummy.mp4"]
G = {"__name__": "agtdefs"}; exec(compile(src, "build_agt.py", "exec"), G); sys.argv = _argv
