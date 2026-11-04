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

### CALI MAN (190 BPM, Oct 26): `build_cali.py`, `VARIANT=A|B`

- Window: song bars 31–44 (0:37.91–0:55.59). Bar 32 is silent, the drop is bar 33 (0:40.45), and the breakdown at bar 41 is melody + perc loop only, so it uses arrangement crops (`MACRO_T` melody rows, `MACRO_P` perc-loop rows, `MACRO_W`).
- **Pattern-mode recordings** (808 first/second 4 bars, clap, hats): FL loops the pattern from 1:01, so song = clip + Dp + B0 + 32 bars (the pattern plays from song bar 33). Each recording's own audio matched the master bar by bar (corr 0.8–0.93) before use.
- **Frame-exact clock OCR** (`cali_clock2.py`, every 3rd native 60 fps frame) beats the 0.2 s OCR + `fps=` filter. OCR misreads "4:xx" as "40x", which looks like gaps; merge those runs.
- **Follow-mode piano rolls** (hats, kick): the playhead is parked at x≈864 and the notes scroll left, and FL only redraws every 2–3 OBS frames. `cali_validate2.py` detects note starts arriving at the parked playhead. The kick roll is two long sub notes, so its hits are entered by hand.
- The hard-cut-on-hit rule only counts hits visible in the shots either side of the cut, and the forced drop zoom punch is kept.
- `VARIANT=C` (drums focus, Jack's request): 808 / clap / hat / kick rolls back to back; arrangement only where no roll recording covers the music (bar 31, the silent bar, the drum-out at bar 40 and the breakdown); no mixer shots. A hat roll whose notes have run out reads as an empty grid, so place hat shots right before a run of hat notes.

### DOPE BOY (190 BPM, Oct 27): `build_dope.py`, `VARIANT=A|B`

- Window bars 7–24 (0:07.66–0:30.39), drop at bar 9 (0:10.18). Face: MVI_5558 (Canon, landscape 1920x1080), cropped 1458x1080 → 1080x800 proxy (`face_dope.mp4`). Same outfit for the whole edit, so only use one face clip per edit.
- **Light FL theme**: clock OCR needs a non-inverted threshold (`dope_clock.py`); notes are orange (R>200, B<115), the playhead core is BGR≈(92,148,191) (`dope_scan.py`).
- **Smooth-follow piano rolls** (snare, 808): notes scroll left at px_per_bar/BAR and play at the keyboard edge (x≈355), where FL draws the playhead. Hit times = per-note predictions clustered (`dope_validate.py`); median error vs master snare onsets ≈ −10 ms. These views get `pic=0` (no OBS picture lead) because the hit position was measured from the picture directly.
- Snare + 808 repeat exactly 8 bars later (band/pitch correlation ≈1.0), so `SNARE8` / `808_8` reuse them in phrase 2.
- "Putting in clap" only has the 4 claps that were already drawn; it's used as one tight (2×) shot around a clap hit. The clap offset needed −50 ms vs the clock-derived D (checked on the output frames).
- **Mixer / plugin shots (Jack's feedback on Dope B):** frame the mixer on the channels whose meters actually move (measure orange meter pixels over the shot; here master, 808, kick, snare, clap), never a generic strip. Show a plugin window (Smooth Operator) whole, in the split panel (1080x1120 fits a wide window), not cropped into a 9:16 sliver. Fixed views need `xmin=0`: the default `xmin` (745) silently pushes the crop to the right.

### ARRESTED (100 BPM, Oct 29): `build_arr.py`, `VARIANT=A|B`

- Window bars 3–16 (0:04.88–0:38.48), drop at bar 9 (0:19.28); bars 15–16 repeat bars 9–10, so 808 / hat pattern / open hat views are reused 6 bars later (`*_R`).
- `arr_locate.py` finds where each recording's own audio sits in the master (all 0.95–0.99 except the Reso bass line, which isn't in this beat's arrangement, so it isn't used).
- **Follow-mode hats:** FL redraws every ~3 OBS frames and the 16th notes are ~70 px apart, so frame-to-frame shift tracking aliases. Take the scroll speed from the grid (px per beat ÷ beat length) instead. Every hit then lands on the 16th grid.
- Clap / 808 / open hat / rim are "putting in" clips: `arr_validate.py` lists the notes not drawn yet (`miss`), and the plan avoids them.
- **Mixer rule in practice:** a mixer shot during the intro or the break shows flat meters. Only place mixer shots where the drums framed in them are playing. Smooth Operator was reused from the Dope Boy session as a non-synced cutaway (it's a spectrum, so there's no visible timing to break).
- Intro with only the melody: open on the wide arrangement, not a tight crop of empty tracks.
- **Arrested fixes (Jack's feedback):**
  - Show the MIDI rather than an arrangement clip sitting at the bottom of the frame. The "Reso Baseline" recording *is* the PayPal [Bass] pattern (same notes, pattern mode). It sits at song bars 5–8 (found by matching its note hits to master onsets), so it covers the 0:05–0:19 bass.
  - The scratch fill is an arrangement page view whose playhead is hidden behind the waveform, so the old shot never moved. Detect the playhead in an empty track below (`ph_rows`), ride it (`roll` + track zoom), and only use the stretch where it's on screen.
  - The lazer channel-rack shot is framed tight on the Zap 3 [Perc] row where he puts it in (clip ~5.3–6.0 s), with a gentle push and no punch zooms.
  - Check every roll shot starts near a real note. A 2-beat lead-in of empty grid before the first rim read as dead, so the previous shot was lengthened.

### RACK ON TOP (144 BPM, Oct 31): `build_rack.py`, `VARIANT=A|B|C`

- Window bars 7–40 (0:10.02–1:06.69), 56.7 s. Each variant uses a different single face clip (A MVI_5561 yellow tee, B MVI_5558 navy, C the GLOCK cap clip), so there's never an outfit change mid-edit. `FACE_SEGS` were hand-labelled for MVI_5561. If a long edit needs more clean face than one clip has, clean stretches get reused (second pass, offset).
- Clock OCR: the small "B:S:T" label sits inside the wider clock crop, so mask it. OCR misreads split runs (e.g. 37.83 vs 27.83 = a "3" read as "2"), so merge runs with the same offset by hand.
- One picture offset for every view (song = clip + Dp + 0.02), fitted from note crossings vs master onsets. Every view lands within ~25 ms.
- The 808 phrase at bar 37 repeats bar 29 (pitch correlation 1.0), so it's reused there (`*_R`).
- Mixer: meters measured per strip. Only master, hats, samples and 808 move, so both mixer framings centre on those.
- Hi-hat roll: don't centre on the crossed row alone. The F#4 rolls sit below the C5 eighths, so keep both rows in frame.
- Before the drums enter, arrangement crops on the drum rows show empty tracks, so use sample/bongo rows there.
- Crop `xmin=462` keeps the piano keyboard strip out of tight roll shots.

### PACKMAN (200 BPM, Nov 1): `build_pac.py`, `VARIANT=A|B`

- Window bars 9-40 (0:09.66-0:48.06). Pattern 4 intro, bar 16 drop-out, pattern 3 drop at bar 17, pattern 2 at bar 25, pattern 1 at bar 33.
- New FL layout (browser docked left): clock crop `140:50:925:8`, with the small B:S:T label masked.
- Jack recorded pattern 2 in **pattern mode**: the clock reads pattern time, so song = clip + Dp + 24 bars (confirmed by audio location and note hits, right after pattern 3 as he said). Several other recordings were also pattern-mode loops, so locate every recording by its own audio first.
- **Smooth-scroll rolls** (hi-hats, 808 pattern 3, bell full) play at the keyboard edge (x≈450). Take px/bar from two ruler numbers on screen, then fit only the play x. Phase-correlating the ruler strip locks onto the static toolbar, so don't use it. Wide 808 notes have no visible edge at that zoom, so their hits come from master sub onsets inside the covered stretch.
- Pattern 1 (bars 33-40) has no roll recording and doesn't reuse the pattern 4 hats (57% match), so it uses the arrangement view, the face shots and Smooth Operator.
- Face: the new copy_99ED92DA_1.MOV is a CapCut export of the same take as GLOCK (mirrored, with a CapCut end card after 29.2 s). hflip it and trim before 29.2 s.
- **Packman revision (Jack's feedback):** "a lot of macro shots towards the end", and 0:18-0:20 "not focused on anything" (a lazer roll cut in before its first note, so it was an empty grid). Fixes:
  - Every roll shot must open within ~1 beat of a visible hit.
  - When a section has no recording, test every recorded view shifted by whole phrases against the master (per-hit, <30 ms) and reuse the ones that repeat: pattern-4 snare +24 bars (6/6), lazer +8 (4/4), pattern-2 snare +8 (only between its undrawn notes), pattern-4 open hats +24 (windowed).
  - Arrangement shots are now only the wide entry/exit of a section.

### WALK THIS WAY (98 BPM, Nov 1): `build_wtw.py`, `VARIANT=A|B`

- Jack asked for about 30 s, so the window is bars 21–32 (0:49.00–1:18.39, 29.4 s). The drums come in at bar 21, there's a fill in bar 28, and the drop is at bar 29. Face: the cap clip from Packman (`face_pac.mp4`).
- **Beige FL theme.** For the clock OCR, use an Otsu threshold (the fixed threshold returns nothing). Crop `140:50:925:8`.
  - Playhead: the red column (R−G > 45, R > 200), taking the median column.
  - Notes are either brown boxes or thin dark bars (hats, open hats and rims). Mask both.
  - Drop any entries in the velocity lane (y ≥ 880).
- **Hybrid page → follow rolls.** The playhead moves across the page, then parks at x≈926 while the notes scroll. `wtw_validate2.py` handles each stretch differently:
  - Page stretches: playhead crossings with an interpolated playhead x.
  - Parked stretches: hits predicted from the median page-mode playhead speed.
  - Follow-mode estimates are noisy (±50 ms, and some are spurious). For checks, cluster them and snap each cluster to the master onset in that instrument's band. `wtw_hitsheet.py` does this.
- **The drop (bars 29–32) is an exported drum bounce.** It's the "Drums exported" audio clip in the playlist, so there's no MIDI for it. `wtw_reuse.py` shifts every validated hit by 4 bars (`*_R`) and 8 bars (`*_8`). It keeps a hit only where the master's onset in that band, within ±35 ms, is at least 0.6× the original's:
  - Rim: repeats exactly at +8.
  - Hats: repeat at +8 in 35.5–38 and 44.8–46 (beats from bar 21). Elsewhere the hat at bar 31 beat 0.7 is missing.
  - Open hat at +8: its bar 32 note doesn't sound, so it isn't used.
  - Toms: don't repeat in the drop.
  - Bar 28 (+4): hats, rim and the fill tom all match.
  - So the drop is rim and hat rolls with the face every 2–3 shots, and only short waveform, guitar and plugin cutaways in between. That avoids the macro run near the end that Jack flagged on Packman.
- **Sync check on the output.** `wtw_rendercheck.py` tracks the rendered notes against the rendered playhead and compares the extrapolated crossings to master onsets. Watch for two things:
  - Back-to-back 16th notes merge into one box.
  - The playhead glow hides the note for ~2 frames, so "note under the glow on the heard frame" is the real test.
  - Source and output are both 30 fps, so a shifted reuse can land ±1 frame (≈33 ms) differently from the original.
- Open-hat notes are pink and thin, so in a tile sheet they look like they sit right of the playhead. Check at full size before "fixing" them.

### MOVE (143 BPM, B minor, Nov 2): `build_move.py`, `VARIANT=A|B`

- Jack asked for 0:17–0:49. The window is bars 11–28 (0:16.8–0:47.0, 30.2 s). Bar 29 restarts the drop note for note, so ending there loops cleanly; going on to 0:49 would cut mid-phrase.
  - Bars 11–12: sample intro plus the fill/riser, shown on the arrangement view.
  - Bar 13: the drop.
  - Bars 21–28: repeat bars 13–20 exactly.
- **Purple FL theme.**
  - Recordings are 60 fps OBS. Clock crop `140:50:925:8`.
  - The playhead and the notes share the same pink-purple (B > 190, R > 150, 55 < G < 130). The playhead is the only column covering more than 55% of the grid rows.
  - Roll crops clamp to y 185–902 (`ymin` / `ymax`) so the title bar, ruler and velocity lane stay out.
- **Every recording plays in song mode at its real position.** song = clip + Dp + 0.02. Audio location agreed with the clock, except New Jazz, which correlates best at other phrases because the pattern repeats.
  - Spinz 808 and the snare: page mode, then follow mode.
  - Sample chops: one slice per beat.
- `move_validate.py` tracks each note across frames using its velocity, so follow-mode scrolling doesn't break the track. It fits note x minus playhead x and extrapolates where that reaches 0. Results against master onsets:
  - spinz −12 ms
  - phantom −20 ms
  - snare +6 ms
  - chops 0 ms
- **Note ends that FL splits off at the playhead** (the box edge sits 1–7 px right of it) are ignored, so they don't count as false hits.
- `move_reuse.py` shifts every picture hit +8 bars and keeps it only where the master has it. Kept: 9/9 spinz, 4/5 phantom, 9/9 snare, 28/29 chops. So bars 21–28 reuse the drop recordings (`*_8`).
- **Clap recording:** the clap notes are 5 px ticks at this zoom and don't read in a 9:16 crop, so there are no clap shots.
- **Mixer pass:** it's at song bars 44–63, which is the third drop and identical to bars 13–28, so it's mapped back by −36 bars. Strips 1–10 are framed: chops, loop, percloop, both 808s, fill, FX, percloop 2, clap, snare.
- Smooth Operator is from this session, synced. Its whole window sits in the split panel.
- **Move revision (Jack's feedback, Nov 2):**
  - **A 0:06–0:08:** "808 note isn't always seen". That stretch of the Spinz recording is him still putting notes in. The Spinz pattern repeats every 4 bars, so `SPINZ_M4` shows the follow-along part 4 bars early, framed wider so the incoming note is on screen from the cut.
  - **A 0:13:** an 808 shot right after the sample slices was "kind of boring footage". Replaced with a tight mixer framing (`MIXER_T`, strips 3–10). B uses a Smooth Operator cutaway there (`PLUGIN_F`, non-synced, the spectrum has no visible timing).
  - **B:** "from 0:16 great, before that clunky". B now uses A's opening 6 s, which Jack liked. A from 0:15 was "flawless".
  - There's no Soft Clipper footage: it's only listed in the mixer inserts, never opened.

### NO COMPETITION (179 BPM, F minor, Nov 3): `build_nc.py`, `VARIANT=A|B`

- **Window:** bars 25–48 (0:32.2–1:04.4, 32.2 s), Jack's "golden" 0:32–1:04, ending where the beat breaks down at bar 49.
  - Bars 25–32: reso bass, kick, hats and melody.
  - Bar 33: drop (808 pattern 7).
  - Bars 39–40: upsweep riser.
  - Bar 41: second drop.
- **FL layout:** same purple theme as Move, but the clock sits further left. Crop `160:50:902:8`, label mask `c[:18, 126:]`.
  - The two "808 follow along" uploads were byte-identical.
  - Each recording plays in song mode at its real position (audio location matched the clock). Picture offset +0.03 s.
- **Phrase reuse:** `nc_reuse.py` checks every hit at the shifted position against the master.
  - Kick: repeats +4 bars, and again at +16/+20 (it rests in bars 33–40).
  - Hats: repeat +4 and +8 (they stop at bar 38.5).
  - 808 (page and follow views): repeat +8 for the second drop.
- **Cutaways:**
  - Smooth Operator on the master, synced, with the whole window in the split panel.
  - MJUC jr compressor on the melody bus, VU needle moving.
  - Melody arrangement and the pre-drop arrangement, both with the playhead parked at the recorded x.
  - Soft Clipper sits in the master chain but was never opened.
- **Camera:** the hi-hat roll is zoomed (one C5 row, notes about 350 px apart), so the camera's playhead smoothing was lightened from 0.7/0.3 to 0.3/0.7. With 0.7/0.3 it lagged and one hat hit landed at the panel edge.
- **Variant C (Jack, Nov 3: "without the first 8 seconds, slightly different edit"):** bars 31–48 (0:40.3–1:04.4, 24.2 s). The drop lands 2.7 s in, after a reso → face/arrangement → kick run-up. `START_J` and `NBARS` now depend on `VARIANT`, and `DROP_J` / `TRANS_FORCE` are derived from `START_J`.
- **Variant D (Jack, Nov 3: "not similar in any way ... fewer jump cuts if it is captivating"):** bars 25–48 with 18 shots instead of 37.
  - It opens on the face split (A–C open on a roll).
  - Long holds where the picture carries itself: 2-bar reso and kick rides, the 808 follow-along phrase scrolling past the parked playhead for ~2.5 s, a 2-bar melody split, and a longer tilted riser.
  - Cuts come only on phrase changes. Faces alternate every 1–2 shots.
- **D revision (Jack, Nov 3):**
  - The melody-arrangement split at 0:07–0:09 was "a boring shot, not really showing what is going on". It is now hats then reso rolls; the second melody split at 0:17 was swapped for the 808 page roll too.
  - At 0:25 the cut from the 808 to the kick broke a shot he liked. It now stays on the 808 page view for the whole phrase (4 s), and the kick and compressor shots after it were dropped.
- **Rack D/E (Jack, Nov 3: "variants of rack C in the style you've just done"):** the same face clip as C. These use the No Competition D style: 23–26 shots instead of 49, long holds on rolls with notes landing, few arrangement shots, and the 808 held through whole phrases.
  - **Drop 2 coverage:** `rack_reuse.py` tests every recording at every phrase shift. The 808 and 808B repeat +4 bars, and +8 bars into drop 2. Drop 2 changes the first bar of each 4-bar 808 phrase, so the unheard notes (beats 108.5 110.5 124.5 126.5 from 0:10.02) are kept off screen. In C, drop 2 was mostly arrangement shots.
  - **Bongo:** the bongo in the split panel shows its velocity lane. The bongo at beat 24.5 is still being drawn, so it reads as an empty grid.
  - **Open hats:** one thin note per bar reads empty even framed wide. Use them only right around their hits.
  - **Hat roll:** paused before clip 4.9 s.
- **Rack D/E revision (Jack, Nov 3):**
  - At 0:28–0:30 "the midi pauses": the hat recording stops scrolling at clip 13.6 s even though the clock run goes on to 14.3. The hats now end at beat 70, and a split drum-arrangement shot bridges into the drop.
  - Two hat shots back to back (0:21, 0:24) showed the same MIDI. The second one is now the compressor (D) or Smooth Operator (E).
  - **QA:** every render now gets a frozen-picture scan (consecutive near-identical frames). The clock alone does not catch a recording that stopped scrolling.
- **Rack D v3 (Jack, Nov 3):**
  - The zoom-punch plus shake at 0:03 felt juddery, so D and E now hard-cut on the bar-9 bass downbeat.
  - The Smooth Operator cutaway at 0:15 was not synced (it is from the Dope Boy session), so it is now a synced arrangement split.
  - At the drop, D follows the 808 follow view full-screen for about 3 s (0:31.7–0:34.8) before cutting.
- **Rack D/E v4 (Jack, Nov 4):**
  - Rough transition at 0:03–0:04: the split-panel bass roll was page-jumping (the held crop re-centres when the playhead leaves the panel). Split panels with rolls now ride the playhead smoothly, as in build_wtw.
  - 0:17 macro shot not engaging: replaced with the open-hat roll in C's framing, on its hit.
  - Borrowed Smooth Operator (E, 0:24–0:27) and compressor (D 0:25, E 0:15) cutaways were not synced: replaced with open hats shifted +4 bars (checked per hit, `OH1_4` / `OH2_4`) and a synced arrangement split.
  - Rack D/E now use no borrowed plugin footage at all; the only plugin/mixer shot is the synced mixer.
- **Rack D/E v5 (Jack, Nov 4): sync and editing rebuild.**
  - **No note editing on screen.** Per-recording `EDITS` windows (bass clip 5.25–6.65 / 8.35–9.75, bongo < 3.5, s808 5.45–7.3, oh2 < 3.85, hat > 13.55) are found by diffing note pixels frame to frame and checked by eye. Coverage = clock runs, with same-offset runs merged (those gaps are OCR misreads), minus the edit windows.
  - **Phrase reuse must be checked on the audio, both ways.** The old per-hit check only asked whether each drawn note had an onset, not whether each heard 808 had a drawn note or the same pitch. Now 4-beat chunks of the 808-band log spectrogram are correlated:
    - drop 1 repeats every 4 bars (corr 1.00);
    - drop 2 (beats 104–136) is a different 808 pattern except beats 120–124.
    - So drop 2 shows the 808 only there, and the rest is mixer, arrangement and open hats, all synced.
  - **Picture offset and camera fixes:**
    - The 808B follow view was a constant 28 ms early: `pic=0.028`.
    - No jolts in D/E.
    - The 808 framing is not re-centred per shot.
    - The 808 page view avoids its page flip (beat 91.5).
  - **QA added:** a frame at every heard 808 onset, from the master and not from the picture hits (`sheets/<v>_808heard.png`).

### KUSH (193 BPM, C minor, Nov 4): `build_kush.py`, `VARIANT=A|B`

- Window bars 1–18 (0:00–0:22.4); the beat is an 8-bar loop (bars 8 / 16 are fills). Face: the cap clip.
- **Pink FL theme** (light grid): playhead core and notes are hot pink — mask R > 175, R−G > 75, 110 < B < 220; clock crop as No Competition (`160:50:902:8`). Several recordings show clock jumps of exactly 2 / 6 bars mid-run (OCR reading "8" as "6", "16" as "10") — merge same-offset runs across them.
- **Coverage built from the audio, not from shifted hits** (`kush_reuse.py` → `cover.json`): every view is (recording, whole-bar shift); a target bar is allowed only if its log spectrogram correlates ≥ 0.97 with the source bar in the bands that instrument lives in (808 rolls: 25–160 Hz; drums: 808 + hi + snare bands; mixer / plugins: all), and only inside clean playback (majority-offset clock runs, minus note-editing windows).
- "Putting in" recordings: the main-808 one is all editing (follow-along recording used instead); the second 808 is drawn until clip 9.9 s.
- Mixer: meters measured — master, 808 #1/#2, 808, clap, countersnare, perc, snare move; MIXER frames master + strips 1–7, MIXER_T tight on the 808 strips; both get the downbeat punch zoom. Soft Clipper framed tight on its window (meters moving), punch on downbeats. Smooth Operator whole window in the split. No jolts.
- **KUSH C (Jack, Nov 4: "keep retention, tell a story by showing the different parts of the beat"):** a story cut.
  - **Hook:** 808 + tight mixer in the first 4 beats.
  - **Chapter 1, the 808:** a 808 ride, then a Soft Clipper punch.
  - **Chapter 2, the drums:** each element in turn, cut on its own hit (snare, perc, second 808, perc 2).
  - **Bar 8 fill:** on the mixer.
  - **Loop return:** a zoom into chapter 3, the mix (open hat, Smooth Operator, clipper).
  - **Recap:** a fast pass through every part on its hit.
  - **End:** a 2.5 s hold on the face over the same 808 it opened with, so the story loops.
- **KUSH v2 (Jack, Nov 4: "blank looking screen 0:09-0:10"):** the 808 follow-along recording runs past its pattern end onto an empty grey page (clip 10.15–11.9 s) before looping. The clock still ticks there, so it looked like clean playback. It is now excluded (`EDITS["m808f"]`), and the A/B/C slots that used it now show the clipper, the mixer or the next-cycle 808.
  - The sparse second-808 view is framed wider, so its next notes are always on screen.
  - **QA added:** a per-shot sheet (4 frames per shot) to catch empty grids that a hit sheet misses.
