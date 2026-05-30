# Screenshot → Mockup Thumbnail (MCP) — Design

## Goal

Enable manual screenshots (1–3 images) to be turned into a share-ready thumbnail that embeds the screenshot(s) inside a device mockup (laptop/phone), styled with brand colors, then uploaded to Supabase Storage and stored as `platform_contents.thumbnail_url`.

## Scope (MVP)

- No app runtime automation (no Playwright).
- User provides local screenshot file paths.
- Server uploads screenshots + generated thumbnails to Supabase Storage bucket `image` using folder paths:
  - `screenshots/...`
  - `thumbnails/...`
- No new DB tables. No schema change required.

## Tools

### 1) `upload_screenshot`

**Input**
- `file_path: str`
- `object_path?: str` (default `screenshots/<uuid>.png`)

**Output (JSON)**
- `url: str` (public URL)
- `object_path: str`
- `width: int`
- `height: int`

### 2) `generate_mockup_thumbnail`

**Input**
- `screenshot_urls: list[str]` (>=1)
- `device?: "auto" | "laptop" | "phone"` (default `auto`)
- `size?: str` (e.g. `1200x627`, `1600x900`, `1080x1080`)
- `headline: str`
- `subheadline?: str`
- `brand_hex?: list[str]`
- `upload?: bool` (default `true`)
- `object_path?: str` (default `thumbnails/<uuid>.png`)

**Behavior**
- Download screenshots.
- Compose a canvas:
  - gradient background from `brand_hex` (fallback: dark neutral)
  - device frame (simple rounded rectangle + shadow)
  - screenshot embedded into “screen” area
  - headline/subheadline text area
- Upload final PNG to Supabase Storage (if `upload=true`).

**Output**
- Text JSON meta: `{ thumbnail_url, object_path, size, device }`
- Image preview content

## Non-goals

- Login/auth flows and routing to specific feature screens.
- Pixel-perfect device frames and typography (acceptable “SaaS-style” mockup).
- Per-platform size presets beyond explicit `size` input.

## Risks / Mitigations

- Missing fonts: use default PIL font.
- Large screenshots: downscale before compositing.
- Supabase config missing: hard fail with clear env error.
