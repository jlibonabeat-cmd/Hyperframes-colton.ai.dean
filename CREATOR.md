# Creator Profile — prodbyjlib (Jack Jones)

Your identity, positioning, and workflow for video work in this workspace. **Read this before every video task** (alongside `PREFERENCES.md`). Use it for on-screen text, niche framing, and workflow assumptions.

> Set up on 2026-10-19 via `/setup`. Re-run `/setup` or edit this file by hand any time.

## Identity

- **Name on-screen**: prodbyjlib (producer alias; real name Jack Jones)
- **Instagram**: @prodbyjlib
- **TikTok**: — (not provided yet; ask before putting any TikTok handle on screen)
- **YouTube**: — (not provided yet)
- **X / other**: — (not provided yet)

When putting handles on-screen (lower-thirds, outros, end-cards), use the platform-appropriate handle.

## Platform priority

**Reels-first** (Instagram Reels). Default aspect ratio: **9:16 vertical, 1080×1920**, 30 fps. Cross-posting to TikTok / Shorts is a bonus, not the target.

## Content niche

**Music production** — music producer posting as @prodbyjlib. When planning a new video, assume this niche unless told otherwise.

## On-camera mix

**Both** — some videos are face-cam, some are faceless.

- **Face-cam** → use `/short-form-video` face-mode choreography (BOTTOM / FULLSCREEN modes).
- **Faceless** → motion graphics + AI TTS narration (`npx hyperframes tts`) or screen-recordings.

## Workflow — division of labor

**Mode chosen: build from scratch with the assistant** → motion graphics, TTS narration, screen-recordings assembled together.

(Alternative mode, if this ever changes: you record + pre-edit your own speaking video and the assistant only builds the visual layer on top without cutting your audio or pacing.)

## Brand identity

Default dark palette with a mint-green accent (matches the FL Studio notes and playhead in your edits) and system fonts, set during /setup on 2026-10-19. Values live in `assets/brand-tokens.css`; change them any time. No logo yet.

## Inspiration creators

- **@jsewellbeats** — primary editing reference: crisp, fast, beat-locked face-cam + DAW edits. Fingerprint: [`_reference/creator-library/jsewellbeats/style-distilled.md`](_reference/creator-library/jsewellbeats/style-distilled.md) (medium-low confidence; built from screen recordings).
- Own edits used as the style baseline: GLIDE BEAT EDIT and Don't Tell Em (both 9:16, face-cam over FL Studio).

Studied creators live in [`_reference/creator-library/`](_reference/creator-library/). Paste a TikTok / Instagram Reel / YouTube Short URL and run `/study-creator <url>` to add one. At build time, name a creator — "build this like @handle" — to apply their style fingerprint to the visual layer.

## Posting cadence & length defaults

- _Cadence_: _(TBD — set during /setup)_
- _Default length_: 15–45s short-form vertical is a common sweet spot. Adjust per video.
- _Default fps_: 30 (matches TikTok + Instagram defaults).

## Project slug convention

**Topic-only, kebab-case.** Each video gets a new subfolder under `video-projects/<topic-slug>/`. Examples: `ai-agents-intro`, `prompting-101`, `tool-of-the-week`. Keep slugs short and descriptive — scannable at a glance in `ls video-projects/`.

---

*This file is updated as your creator profile evolves. Major changes (new handles, platform shifts, workflow changes) can be edited here directly.*
