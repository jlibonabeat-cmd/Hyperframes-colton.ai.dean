# YOSHI EDIT pipeline (clone / disappearing, after @prodyoshi01)

First test: Something2say (144 BPM), Nov 22. Inputs: IMG_9930.mov (empty desk), IMG_9931.mov (Jack at the desk), beat.

1. **Colour:** iPhone clips are HLG HDR (bt2020 / arib-std-b67). Convert both clips the same way with
   `zscale=tin=bt709:t=bt709:pin=bt2020:p=bt709:min=bt2020nc:m=bt709:r=tv`, i.e. read the HLG signal as its
   backward-compatible SDR view. This matches the phone's own look. Never mix converted and unconverted clips.
2. **Align:** the two clips differed in size (3114x1752 vs 3200x1800, a 2.6 % zoom difference).
   - Fit an ORB + RANSAC homography on the background only, with the person area masked.
   - Warp Jack's take onto the plate, then match per-channel gain/offset to the plate's wall and desk, refitting every second (iPhone auto-exposure).
3. **LED strip:** it cycles colours in both clips, so it changes on every pop. That's accepted as part of the look.
4. **Pop states:** the empty room plays in song time; each appearance plays on from its chosen moment.
   - Cuts snap to the nearest frame of the snare / kick / pickup hits.
5. **Clones:** need takes with him in different spots. With one standing spot only the disappearing half of the style works.
   - The matte prototype (Lab difference vs a median plate, morphology, desk-strip gap bridged) is good enough for two non-overlapping clones.
