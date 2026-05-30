## Goal

Improve thumbnail quality by letting the image model handle creative layout/background while keeping the workflow “1 prompt → MCP tools → saved to Supabase” stable.

## Context

Current `launch_linkedin_from_screenshots` generates a deterministic PIL mockup (gradient + device frame + text). Output feels templated.

## Proposed Change

Keep the tool signature the same, but change internal rendering:

1. Read local screenshot file bytes from `screenshot_paths[0]`.
2. Use OpenAI Images Edits (`/v1/images/edits`) with guardrails prompt to produce a creative marketing thumbnail background that incorporates the screenshot (no gibberish text).
3. Resize/crop the AI image to the target size (default `1200x627`).
4. Overlay headline/subheadline using existing PIL text rendering (wrapping + auto-fit).
5. Upload result to Supabase Storage and save draft to `launches` + `platform_contents` (same as today).
6. Fallback: if the image edit call fails, fall back to existing `generate_mockup_thumbnail` (gradient/AI background).

## Guardrails Prompt (Thumbnail)

- Modern SaaS marketing thumbnail, premium, high contrast, clean.
- Must keep the provided UI screenshot recognizable and readable.
- No new logos, no watermarks, no random UI text, no people.
- Leave clean negative space on the left for headline text overlay.

## Copy Prompt (LinkedIn)

Maintain the “indie hacker” voice but enforce:

- 1 hook line
- 3 short bullets (feature → benefit)
- 1 short CTA (no link)
- 0–2 emojis
- up to 3 hashtags at end

## Compatibility

- No changes to MCP tool names/args.
- Works with existing Supabase schema.

## Validation

- Run `scripts/e2e_macro_linkedin_from_local.py` using `quote-card/images/app/image.png`.
- Verify: draft saved, thumbnail URL renders, screenshot remains recognizable.
