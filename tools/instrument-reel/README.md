# Instrument reel (Jack playing guitar/synth from phone angles, simple cuts)

Built on "Catch Me" (90 BPM, Oct 2026). The only output audio is the master mp3. Each phone clip's own audio is used just to sync it.

1. Proxy each phone clip to 1080x1920 / 30 fps with its audio (`ffmpeg -vf scale=1080:1920,fps=30`).
2. `cm_sync2.py` / `cm_sync3.py`: chroma + onset matching of each clip against the master. Jack loops the riff while filming, so each take has one offset D (song = clip + D), and it only matches modulo the riff length (4 bars here). Takes drift by ~30–80 ms over 20 s, so align per shot.
3. `cm_plan2.py`: cut points on chord changes (half bars here), angle order (wide → hands → side → front ...), and for each shot the best clip position **constrained to that take's beat grid** (`take_D`, ±0.12 s). Without the constraint it happily picks positions a beat out.
4. Re-measure each shot's residual lag (clip audio vs master) and shift `c0` by it. Every shot ended within ±2 ms.
5. `cm_render.py`: exact frame counts per shot, hard cuts, master audio with a tiny fade at the end. No zooms, whips or effects.
6. QA: tile every frame. No talking, no lens looks, no camera handling.
