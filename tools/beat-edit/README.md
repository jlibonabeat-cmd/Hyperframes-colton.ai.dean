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

1. **Face-cam:** label the whole phone clip at 2–4 fps and keep only stretches where Jack looks at the screen. No eye contact with the lens, no talking (mouth moving) and no reaching for the phone (`FACE_SEGS`).
2. **Sync per shot:** for every piano-roll shot, render the exact output frames at each heard hit (from `hits.json`) and check that a note sits on the playhead in each one. A recording can be in sync but not show the part you hear: the main-snare roll only has the backbeat, so cut away before the other snare layers come in.
3. **Crops:** never letterbox or fit-to-width. Every view fills its panel.
4. **Every frame:** tile all frames (15x5 per sheet) and look at all of them before sending.

### JAIL v5

- Hat recording check: compare the recording's own hat rolls (shifted by D) with the master's rolls. The hat clip has a roll at song 25.31 that the master doesn't have (it was edited later), so the hat shot is limited to 24.21–25.23, where every heard hat has its note on screen.
- Camera moves: `track` (1.55× macro riding the playhead), `tilt` (playlist rotated ±7°→±3° with a slow push, clamped inside the crop), `push`/`pull` on split shots, `mixer` punches, and `impact` on the opener.
- Transitions cycle whipH / zoom / hard / whipV (motion-blurred). The drop is a zoom punch with a soft flash, and the cut into the silent bar is hard.

### JAIL with new face footage (IMG_9481, Oct 23), `build_jail2.py`, `VARIANT=A|B`

- New phone face-cam: front-camera clips are mirrored (the cap text reads backwards), so `hflip` them. Crop band FACE_Y0=460.
- `REACT_T`: a "stank face" moment pinned to the shot where the 808 comes in.
- 808 focus: the 808 enters at bar 26 (j=9), one bar after the drop. Variant A puts the 808 full-screen there. Variant B has him reacting on top with the 808 below, then full-screen 808. Full-screen 808 shots use a wider track zoom (1.12→1.22) so the whole 808 phrase reads.
- `TRANS_FORCE` sets the transition into the drop and into the 808 shot.

### Colour-graded phone clips are HDR (HLG). Convert them properly

iPhone exports can be HEVC 10-bit BT.2020 HLG (`color_transfer=arib-std-b67`). Decoding them straight to 8-bit treats them as SDR: the grade washes out to pale and flat, and Jack's grade disappears. Always check `ffprobe ... color_transfer` and tone-map:

`zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=gbrp`

Then, if needed, match it to how it looks on his phone. Fit a per-channel curve against a screenshot he sends of the same frame and apply it as a `.cube` (`lut3d`). Compare a before/after frame before sending.

### GLOCK ON MY HIP (149 BPM, Oct 25): `build_glock.py`, `VARIANT=A|B`

- Window: song bars 4–23 (0:06.48–0:38.69). The intro (bars 4–7) only has the arrangement view, the drop is at 0:12.92 and phrase B at 0:25.80.
- **Reusing a recording 8 bars later:** snare and hat patterns repeat exactly in phrase B (band correlation 0.97–1.00), so those views get a second segment at D + 8 bars. The 808/kick and melody change in phrase B, so they are not reused.
- `playhead.lo` / `xmin` per view: these piano rolls start at x≈100 and the arrangement playhead sits at x=493 (follow mode). The default crop clamp (745) cut the arrangement playhead out, so arrangement views need `xmin=330`.
- "808 pattern 2" never shows a playhead (the pattern isn't playing in that recording), so it isn't used. The "putting 808 in" recording is only valid from bar 11, after the notes are placed.
- Soft Clipper gets the mixer punch-zoom treatment.

### GLOCK B → B2: what was wrong and the fix (Jack caught it)

- **Notes outside the crop:** the sample-chops notes sit at the bottom of the roll (y≈750–890) but the crop was centred on y=650 and zoomed in, so the shot showed an empty grid. Roll shots now take their vertical centre from the notes the playhead actually hits in that shot (`validate.json` → `cy`).
- **Playhead crossing empty space:** these rolls are sparse (snare ≈1.5 notes per bar, kick ≈1 per bar, scratch 0–1, 808 part 1 1). A 1.6 s shot of a sparse roll while the drums bang reads as "out of time". Fix: `glock_validate.py` lists every moment the playhead hits a visible note (in song time). Roll shots are now 2–4 beats long and placed around those hits, with the arrangement, Soft Clipper and face covering the rest.
- **"Putting in" recordings:** notes are added during the recording, so some heard hits have no note yet. `glock_validate.py` also flags crossings of final-pattern notes that aren't drawn yet ("missing"). The plan avoids those moments.
- Don't judge sync with band-pass "snare/kick" onsets from the full mix: the sample and 808 pollute those bands. Judge it from the playhead/note crossings plus the per-bar match between the recording's own audio and the master (`glock_barmatch.py`). Then render the output frames at every crossing (±1 frame) and look at them.
- Jolts are never placed within 0.25 s of a note hit.

### GLOCK C / D variants (Oct 25): `VARIANT=C|D`, `SHAKE=1`

Jack approved B2 and asked for more variants: same footage, different parts at different times, even more on time, J Sewell transitions with a bit of shake.

- **Transition shake** (`SHAKE=1`, default on): after every whip / zoom transition, a damped 7-frame shake (18 px for whips, 26 px for zooms, decaying, seeded RNG 149), each frame motion-blurred along its path. Hard cuts get none.
- **Hard cut on hits:** any cut within 3.5 frames of a validated note crossing becomes a hard cut, otherwise the hit lands on a whip-smeared frame.
- **HAT coverage limit:** hat notes only exist up to clip 2.95 s; after that the roll is an empty grid. HAT segments end there.
- Variant C leads with hats and snares; variant D is kick/808-led with more face shots. `glock_hitsheet.py <variant>` renders the output frame at every crossing for checking.
