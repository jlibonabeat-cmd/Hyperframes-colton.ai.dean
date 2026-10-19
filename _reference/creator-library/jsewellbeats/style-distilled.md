# @jsewellbeats — style fingerprint

Producer content: face-cam + DAW (FL Studio style), 9:16, crisp and fast, visuals locked to the music. Chosen by the user (@prodbyjlib) as the editing style to match: "really crisp edits", "sometimes shows his face sometimes not", "edits correlate really well with his music and beats", "I want my edits to jump and have similar effects syncopated with my beats."

**Confidence: medium-low on fine detail.** Built from 4 phone-quality screen recordings of Instagram (partly covered by the comments panel), not original files. Trust the structure and cadence; do not trust exact fonts, easings or colours. Details and caveats: `videos/screen-recordings-2026-10-19/analysis.md`. Refresh this file with `/study-creator <url>` once reels can be downloaded, or with clean screen-recordings of single reels with the comments panel closed.

## Signature moves

1. **Layout jumping.** One screen-recording and one face-cam are cut between three layouts, never staying in one for long: split (face on top ~40%, DAW below), DAW full-screen close-up, face full-frame. The layout change is the "jump".
2. **~1 visual event per second**, median gap ~0.7-0.8 s. Holds longer than ~3 s are rare; the longest seen was ~5 s.
3. **Bursty rhythm.** Calm stretches, then 5-10 rapid hits inside ~0.5-1 s on the drops / fills. Plan a few of these per 25 s, not a constant machine-gun.
4. **Macro note shots.** Tight crop on a piano roll with 1-2 notes visible on an empty grid and a thin cyan playhead crossing it. Each note landing is a visual hit.
5. **Smear / blur / pixelation on layout changes** (face blurred mid-cut, horizontal streaks, a mosaic frame), not clean dissolves.
6. **Single-frame-to-1-second cutaways** (a film still, a reaction) used as a punctuation beat.
7. **Talking-head captions:** white, bold, rounded geometric sans, sentence case, 1-3 words per chunk, no box, lower third (~70-77% of frame height).
8. **Comment-keyword CTA** in captions ("comment X and I'll send you the link"), a content habit rather than an edit one.

## Compared with the user's own edits (GLIDE BEAT EDIT, Don't Tell Em)

The user already shares the layout language: split face-cam over FL Studio, full-screen piano roll / plugin close-ups, plugin windows popping in. What differs:

| | @jsewellbeats | @prodbyjlib today |
|---|---|---|
| Distinct cuts per second | ~1.0 (median gap 0.7-0.8 s) | ~0.6-0.66 (median gap 1.0-1.2 s) |
| Rapid bursts | Several per reel | Mostly one at the finale |
| Effects | Smear / pixelate / zoom on most layout changes | Strong but rarer: light-leak flash into the split (Glide ~1.3 s), 1-frame flash inserts (yellow texture ~10.75 s, fire warp ~15.5 s), finale fisheye + RGB glitch + "ERROR" slam (Glide 27.5-29 s), shatter-to-white (Don't Tell Em 35-37 s) |
| Holds | Mostly under 2 s | Several 3-5 s holds on one view (face-cam + piano roll) |

Takeaway: keep the user's finale effects (they are a strength), and add the density: tighten holds, add 1-2 mid-video bursts, and put smear transitions on the layout changes.

## Rules for applying this to @prodbyjlib beats

These are the build-time rules; they override defaults but sit below `PREFERENCES.md`.

- **Measure the beat from the actual audio first.** Extract kick / snare / hat onset times and the BPM from the track, and place cuts and hits on those times (within about one frame), instead of on a fixed grid. Known tempos from the FL projects: Glide 140 BPM, Fast Lane 180 BPM F minor.
- **Syncopate:** land hard layout cuts on kicks/snares, and put the small accents (a zoom nudge, a flash frame, a note highlight) on off-beat hats and ghost notes so the picture bounces against the rhythm rather than just following the downbeat.
- **Punch pulses:** a 4-6% scale pop on each kick that settles in about 120-150 ms with an ease-out; keep it on the DAW/face layers, not on captions.
- **Section changes get an effect, not a plain cut:** smear whip, light-leak or white flash for 1-2 frames, matching the user's existing flashes.
- **Burst on drops:** 5-8 hits across a half bar; then breathe for a bar.
- **Ends big:** distort / glitch / shatter in the last 1-2 s, as in the user's own edits.
- **Face-cam optional per video** (user mixes face and faceless). Faceless versions use the DAW layouts only, with macro note shots doing the work of the face-cam cutaways.
- **Never** hold a single layout for more than ~3 s unless it is the talking-head moment.

## Hyperframes mapping

- Whip / smear: `whip-pan`, `transitions-blur`, `transitions-push`
- Flash frames: `flash-through-white`, `light-leak`
- Glitch / finale: `glitch`, `chromatic-radial-split`, `thermal-distortion`, `gravitational-lens` (fisheye feel), `transitions-destruction` (shatter)
- Pixelate: component `grid-pixelate-wipe`
- Grain over everything: component `grain-overlay`
