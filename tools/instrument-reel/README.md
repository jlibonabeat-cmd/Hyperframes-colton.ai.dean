# Instrument reel (Jack playing guitar/synth from phone angles, simple cuts)

Built on "Catch Me" (90 BPM, Oct 2026). The only output audio is the master mp3. Each phone clip's own audio is used just to sync it.

1. Proxy each phone clip to 1080x1920 / 30 fps with its audio (`ffmpeg -vf scale=1080:1920,fps=30`).
2. `cm_sync2.py` / `cm_sync3.py`: chroma + onset matching of each clip against the master. Jack loops the riff while filming, so each take has one offset D (song = clip + D), and it only matches modulo the riff length (4 bars here). Takes drift by ~30–80 ms over 20 s, so align per shot.
3. `cm_plan2.py`: cut points on chord changes (half bars here), angle order (wide → hands → side → front ...), and for each shot the best clip position **constrained to that take's beat grid** (`take_D`, ±0.12 s). Without the constraint it happily picks positions a beat out.
4. Re-measure each shot's residual lag (clip audio vs master) and shift `c0` by it. Every shot ended within ±2 ms.
5. `cm_render.py`: exact frame counts per shot, hard cuts, master audio with a tiny fade at the end. No zooms, whips or effects.
6. QA: tile every frame. No talking, no lens looks, no camera handling.

## Versions + zoom (v2–v5)

`cm_make.py [name ...]` builds named versions (cut points in half bars + angle order) in one go. Rules:
- Subtle eased zoom (8% in or out, alternating) only on far-away angles (wide room, front) and only on shots ≥ 2 s. Close-ups and short shots stay static.
- Every shot's take is re-checked against the master. If its residual is > 100 ms, that take is rejected and the next best one (or the next angle) is used. All versions end within ±2 ms.
