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

## Rules for applying this to @prodbyjlib beats (corrected after Fast Lane v1–v7 feedback, Oct 2026)

These are the build-time rules. They override defaults but sit below `PREFERENCES.md`. The earlier draft (fast ~0.7 s cuts, 4–6% punch pulses on every kick, big glitch outro) was tested and **rejected** by the user as jarring, so don't bring it back.

- **Footage locked to the song:** read each FL recording at the song time it is showing (FL clock OCR), so notes cross the playhead when heard. Never cut screen recordings together randomly.
- **Pacing:** about one cut per bar in drops (~1.3 s at 180 BPM) and every 2 bars in breakdowns. Cuts land on kick/snare onsets.
- **Face:** split layout, face panel on top at ~40% with wide framing. Never full-frame face close-ups. Face appears regularly (at least every third shot) and ends the video.
- **Motion:** hard cuts, plus a 1-frame-out / 2-frames-in horizontal motion-blur whip on about every third cut and on section changes. **Jolts** on strong kick/808 hits (60–110 px snap with motion blur, settling over ~5 frames, DAW panel only) about once a second in drops. These have to be clearly visible: 24–46 px was too subtle.
- **No** pulse zooms, slow zoom drift, snare shake, chroma flicker or on-screen text.
- **Ending:** a clean cut on the bar line so the reel loops (see @pandoraxx918). No fisheye or glitch outro on a frozen or dead frame.

Frame-level study of his transitions: `videos/testing-my-new-kit/analysis.md`. Pipeline: `tools/beat-edit/`.
