# @jsewellbeats — analysis of 4 screen recordings (2026-10-19)

**Source quality — read this first.** These are screen recordings of Instagram on a 1920x1080 desktop (60 fps), not the original files. The reel plays inside Instagram's modal, partly covered by the comments panel, so only the left ~60% of each reel was visible, at roughly 480x860 px. Several clips are the user scrolling the profile grid, not a reel playing. Everything below is what was visible; fine detail (exact fonts, exact easing, the part of the frame hidden by the comments panel) is not recoverable. This analysis was made by hand from extracted frames (`/study-creator` needs URLs and could not be used).

## What the four recordings contain

| File (upload order) | Length | Content |
|---|---|---|
| 2026-10-19_17-29-02 | 96 s | Profile reels grid scrolling, a few reels opened briefly. Mostly useful for his thumbnail look. |
| ...17-30-43 | 18 s | Modal opens for a moment, then grid. Little reel content. |
| ...17-31-23 | 54 s | Two reels playing: (A) ~26 s, "testing my new kit" / comment-for-waitlist CTA; (B) ~6 s talking-head with captions about Suno Studio. |
| ...17-32-50 | 25 s | One reel playing (~24 s), caption mentions a sample/plugin pack and a "comment PHANTOM for the link" CTA (partly legible). |

## Look and layout

- 9:16 vertical. Dark, high-contrast, teal/blue and orange practical lighting on him; DAW (FL Studio style) in dark theme with a cyan/teal playhead line and red/orange clip colors.
- Three layouts alternate constantly:
  1. **Split:** face-cam on top (~40% of frame), DAW strip below (waveform/clip name or piano roll).
  2. **DAW full-screen close-ups:** piano roll, playlist, plugin windows (soft clipper, Ozone-style EQ) filling the frame.
  3. **Talking head:** face-cam fills the frame, with large captions.
- **Macro note shots:** tight crops of a piano roll where one or two notes are visible on an empty dark grid, with the playhead sweeping through (reel A, ~5-9 s and ~15-18 s). Each note is its own visual hit.
- **Cutaways:** a film still of a woman inserted for about a second (reel A, ~19-20 s) as a reaction/mood beat.
- **Captions (reel B):** white, bold, rounded geometric sans, sentence case, 1-3 words at a time ("has some", "new features."), no box, sitting in the lower third around 70-77% of the frame height.

## Cuts and effects

- About **1 distinct visual event per second** (reel A 1.05/s, reel C 0.93/s), median gap about 0.7-0.8 s. Longest holds were 3-5 s (face-cam talking or a long playhead sweep).
- Rhythm is **bursty**: calm stretches, then flurries of 5-10 rapid hits inside about 0.5-1 s (reel A at roughly 5.2-5.4 s, 10.4-10.8 s, 13.1-14.5 s, 17.3-18.5 s, 23.2-24.1 s). These read as effect flurries rather than separate scenes.
- **Transition types seen:** hard cuts between layouts; blur/smear frames on layout changes (face blurred mid-cut, horizontal red/teal smear streaks); a mosaic/pixelation frame (reel B, ~47 s); quick zoom-in on a plugin or note.
- Because most cuts change *layout* (split to full-screen DAW to face) rather than angle, the edit feels like it "jumps" even with one camera and one screen recording.

## Beat sync — what I could and could not measure

- Automatic measurement of how tightly his cuts land on the beat was **inconclusive**: the numbers came out at chance level, including for the user's own edits where the cuts are plainly visible. The detector was picking up screen-recording noise and glitch bursts, and tempo estimates from 6-26 s windows were unreliable. Do not cite them.
- Visually, the flurries and macro note shots clearly track the music's hits. The usable rule is therefore qualitative: visuals move on the audio's onsets, not on a fixed clock.
- When building, measure the beat from the user's own audio (see `style-distilled.md`).

## Privacy note

The recordings show the user's browser tabs and some account pages around the reel. None of that was recorded or used.
