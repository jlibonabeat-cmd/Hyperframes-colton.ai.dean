# Beat-edit pipeline (FL Studio screen recordings → 9:16 beat-synced reel)

Built on the Fast Lane edit (v1–v7, Oct 2026). Python + ffmpeg + OpenCV, rendered frame by frame, not Hyperframes HTML. The skill `beat-sync-edit` describes when and how to run it. This file covers the mechanics.

## Inputs (one working folder, set `BEAT_EDIT_DIR` or run from inside it)

| File | What |
|---|---|
| `beat.mp3` | the beat (full track) |
| `<name>.mp4` | silent OBS screen recordings of FL Studio, 1920x1080. The FL transport clock (B:S:T) must be visible top-centre |
| `face/<name>.mp4` + `face/info.json` | face-cam clips. Raw face-cam is best; in Fast Lane they were cropped from the user's Glide / Don't Tell Em edits (1620x930) |
| `segs2.json` | per clip: `[clip_a, clip_b, D, spread]` playback segments, where song_time = clip_time + D |
| `hits.json` | kick / snare / hat onsets + strengths for the chosen window (`hits.py`) |

## Steps

1. **Clock OCR → song time.** `clockmap.py` OCRs FL's B:S:T clock every 0.4 s (`crop=138:52:893:2` for a 1920x1080 OBS capture; adjust if the layout differs). Seconds = ((B-1) + ((S-1) + T/24)/16) × 4 × 60/BPM. Steps are 1–16, ticks 0–23.
2. **Segments.** Group consecutive readings where clip_time − song_time is constant: each run is a playback segment with offset D (use the median; drop OCR misreads). Re-check D with exact `ffmpeg -ss` reads (the `fps=` filter is off by ~0.17 s). Trim 0.1 s margins and cut segments around OBS windows, dialogs and view changes (look at the frames!). Write `segs2.json`.
3. **Onsets.** `python3 -I hits.py <START> <DUR>` → `hits.json` (kick 35–130 Hz, snare 1.5–5 kHz, hat 7–16 kHz).
4. **Configure `build_beat_edit.py`:** `START` / `DUR` (bar-aligned, where the recordings cover the song), the `V` views dict (clip, instrument, `roll` or `play`, segments, crop centre y and height), `PLAYHEAD_FALLBACK`, `FACE_CX` / `FACE_ORDER` / `FACE_START`.
5. **Render:** `python3 -I build_beat_edit.py out/name.mp4 [t_from t_to]`. It writes `out/name_events.json` (shot list).
6. **Review** frames (contact sheets plus frame-exact checks around jolts, whips and the ending) before showing anyone.

## What the renderer does (v7 rules, from the user's feedback)

- Every shot reads its recording at `clip_time = song_time − D`, so notes cross the playhead when heard. If no recording covers a stretch, it holds the nearest frame under the face (never a full-frame face).
- Cuts: every bar in drops, every 2 bars in breakdowns, snapped to kick/snare onsets. Extra cuts at coverage edges. Same view twice in a row is merged into one shot.
- Layouts: `full` (DAW), `face` (face panel 800 px of 1920, wide framing, DAW below), `dual` (two DAW views). Face appears at least every third shot, never twice in a row, and on the last shot.
- Crops are held still per shot and only re-centre when the playhead reaches the edge. Zoom is fixed, with no drift.
- Motion (@jsewellbeats style): hard cuts, a sideways motion-blur whip on about every third cut and on section changes, and jolts (60–110 px sideways or vertical snap with motion blur, settling over 5 frames) on strong kick/808 hits about once a second in drops and less in the breakdown. Only the DAW panel moves, not the face. Small 2–3% scale bump with each jolt.
- No text, no labels, no handle. No shake on snares, no pulse zoom, no chroma flicker.
- Ending: clean cut on the bar line (0.12 s audio fade) so the reel loops into the opening drop. No fisheye or glitch outro on a dead frame.

`example-fast-lane/` holds the Fast Lane data (`segs2.json`, `hits.json`, face `info.json`, final shot list) as a worked example.

## JAIL (Oct 2026): the newer, cleaner pipeline. Start here next time

`build_jail.py` is a from-scratch renderer in the @pandoraxx918 / @jsewellbeats style, and the better template. Steps:

1. `jail_clock.py <clips...>`: OCR FL's clock (crop `150:48:904:6` for this OBS layout) and group the readings into playback runs (song = clip + D). OCR misreads 7→1 sometimes, which splits runs. Merge runs with the same D.
2. `jail_xcorr.py <clips...>`: refine each run's D by cross-correlating the recording's own audio with the master (correlation 0.9–0.99). Use D = mean(clock, audio). The recordings' audio is only for syncing; the master mp3 is the output audio.
3. `jail_playhead.py`: map where each piano roll actually shows a playhead. A pattern's playhead only appears while that pattern is playing in the song; outside that, the notes sit still and the shot looks dead. Trim segments to those ranges.
4. Note the view types: `roll` (playhead moves across the piano roll, the crop follows it), `fixed` + `ph` (playlist in follow mode: the playhead is parked at x≈994 and the song scrolls under it), `fixed` + `fxl` (piano roll in smooth-follow: notes stream left and play at the keyboard edge), `fixed` + `fx` (mixer).
5. The vertical phone face-cam (4K, rotated) is proxied to 1080x1920 and an 800-px band is cropped (FACE_Y0) for the top panel. It plays continuously across shots like a real camera.
6. The shot plan is explicit cut bars (`CUT_BARS`). Face every ≤3 shots, never twice in a row, on the silent bar and on the last shot. Whips (multi-sample motion blur) on the drop and ~every third cut, plus a soft flash on the drop. 3 jolts max, motion-blurred along their path, on strong 808 hits away from cuts and outside the last 2 bars.

### JAIL v3 lessons (sync bugs Jack caught)

- **Smooth-follow piano rolls hide the playhead.** The notes play at a fixed x somewhere mid-screen, not at the keyboard edge. Find it from the ruler bar labels at known song times (x_p = label_x + bar_fraction × px_per_bar, mod px_per_bar), then draw FL's playhead there (`draw_ph`). Cropping at the keyboard edge put the hats about 1.4 s out.
- **Use the audio cross-correlation offset, not the clock OCR.** OCR reads about 40 ms late (display latency), and the audio offset keeps OBS's own picture/sound relationship. Check it with `verify2.py`: playhead vs note at each master onset should be within about 1 frame.
- **Skip clips where you re-draw one note over and over.** They never read as in time.
- **"Putting in" clips need the whole pattern in frame** (`fit` mode: pattern area scaled to the panel width on a blurred fill), otherwise the notes being placed fall outside the crop.

### JAIL v4 QA routine (do this before sending anything)

1. **Face-cam:** label the whole phone clip at 2–4 fps and keep only stretches where Jack looks at the screen. No eye contact with the lens and no reaching for the phone (`FACE_SEGS`).
2. **Sync per shot:** for every piano-roll shot, render the exact output frames at each heard hit (from `hits.json`) and check that a note sits on the playhead in each one. A recording can be in sync but not show the part you hear: the main-snare roll only has the backbeat, so cut away before the other snare layers come in.
3. **Crops:** never letterbox or fit-to-width. Every view fills its panel.
4. **Every frame:** tile all frames (15x5 per sheet) and look at all of them before sending.
