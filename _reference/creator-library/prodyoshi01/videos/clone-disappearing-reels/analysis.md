# @prodyoshi01: "clone / disappearing" producer reels — Jack calls it the **YOSHI EDIT**

Source: Jack's screen recording (2026-11-22, 75 s) of four of the reels on Instagram desktop. Liked by @jsewellbeats.
Reels: "Baseline > chords" (purple LED room), "Slow sexy drill" (green LED), "i don't want u" (magenta), and
"I've stopped making music for algorithms..." (daylight).

## The look
- **One locked-off tripod wide of the whole studio**, never moving: no zoom, no shake, no tilt, no transitions.
  - Phone camera, slightly wide-angle, eye-to-chest height, from a corner, floor in the bottom third.
  - Desk / keys / LED strip across the middle.
- Mood comes from the room lighting (coloured LED strips, dark room, or flat daylight), not from a grade.
- Music only: the sample or beat. No talking.
- A short caption in a white or black box at the top: the title, e.g. "baseline > chords". Jack's rule is no on-screen text unless he asks, so leave it off.

## The trick
- Built from an **empty plate** of the room plus takes of him doing things in it.
- Because the frame never moves, he can be composited in and out.
- **Pop in / pop out on the beat:** a hard cut between "him somewhere" and "empty room", or a different pose or position.
  - Empty-room frames between appearances are the "disappearing" part.
  - A few cuts are 2-4 frame ghost dissolves instead of hard pops.
- **Clones:** two (sometimes three) of him on screen at once, each a different moment of the takes. Examples:
  - one in the chair at the desk, one at the keyboard;
  - one lying on the floor, one at the keyboard;
  - two facing each other.
- Each clone appears or vanishes on a musical event:
  - a new layer coming in ("baseline > chords": a new clone when the chords start);
  - kick or snare hits;
  - hi-hat rolls → rapid-fire pops every ~0.15-0.3 s.
- Poses are performative: hands over his face, holding a keytar, nodding, leaning on the desk, walking across, sitting, lying down. They read clearly at phone size because he is big in the frame (full body, a third of the height).

## Pacing (measured)
- Slow sections: a new state every ~1-2.5 s (on bar or half-bar lines).
- Busy sections: bursts of 6-10 pops in 2 s (hats, fills, the drop), then a longer hold.
- Reels run 15-17 s and loop. Each ends on him in the room, or the room empty.

## How we'll build it
1. Inputs:
   - the empty-room plate;
   - one or more takes of him in the same locked frame (same lighting, the phone not moved);
   - the sample/beat.
2. Matte him out of each take: the difference against the plate, then feathered and cleaned (morphology, temporal smoothing).
   - Shadows and light spill come with him: keep a soft matte so the floor shadow stays.
3. Choose 3-6 poses / moments from the takes. Map each to musical events (layer entries, kick/snare, hat rolls) from the beat's onset/section analysis.
4. Composite layers over the plate, each switched on/off with hard pops on those events:
   - single pops in slow parts;
   - stutter bursts on rolls;
   - 2+ clones at the drop / busiest section.
5. 1080x1920, no camera moves, no grade, no text. End on a clean bar line so it loops.
