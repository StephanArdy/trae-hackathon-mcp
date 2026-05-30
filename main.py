import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from mcp import types
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage
from PIL import ImageColor, ImageDraw, ImageFilter, ImageFont

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

mcp = FastMCP("LaunchPilot")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _get_openai_text_model() -> str:
    return os.environ.get("OPENAI_TEXT_MODEL") or "gpt-5.5"


def _get_openai_image_model() -> str:
    return os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1"


def _get_openai_bg_image_model() -> str:
    return os.environ.get("OPENAI_BG_IMAGE_MODEL") or "gpt-image-2"


def _supabase_url() -> str | None:
    return os.environ.get("SUPABASE_URL")


def _supabase_service_role_key() -> str | None:
    return os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def _supabase_bucket() -> str:
    return os.environ.get("SUPABASE_STORAGE_BUCKET") or "image"


def _supabase_headers(service_role_key: str) -> dict[str, str]:
    return {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
    }


def _extract_responses_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for c in item.get("content", []) or []:
            text = c.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    raise RuntimeError("OpenAI returned no text output")


def _openai_responses(model: str, input_text: str) -> str:
    api_key = _require_env("OPENAI_API_KEY")
    resp = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": input_text},
        timeout=120,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        raise RuntimeError(resp.text)
    return _extract_responses_text(resp.json())


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _supabase_upload_png(png_bytes: bytes, object_path: str) -> str:
    url = _require_env("SUPABASE_URL").rstrip("/")
    key = _require_env("SUPABASE_SERVICE_ROLE_KEY")
    bucket = _supabase_bucket()

    encoded_path = quote(object_path, safe="/")
    upload_url = f"{url}/storage/v1/object/{bucket}/{encoded_path}"
    resp = httpx.put(
        upload_url,
        headers={
            **_supabase_headers(key),
            "Content-Type": "image/png",
            "x-upsert": "true",
        },
        content=png_bytes,
        timeout=120,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        raise RuntimeError(resp.text)

    return f"{url}/storage/v1/object/public/{bucket}/{encoded_path}"


def _parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError("size must be like 1200x627")
    return int(parts[0]), int(parts[1])


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _gradient_bg(width: int, height: int, colors: list[str] | None) -> PILImage.Image:
    if colors and len(colors) >= 2:
        c1 = ImageColor.getrgb(colors[0])
        c2 = ImageColor.getrgb(colors[1])
    else:
        c1 = (15, 23, 42)
        c2 = (30, 41, 59)

    base = PILImage.new("RGB", (width, height), c1)
    top = PILImage.new("RGB", (width, height), c2)
    mask = PILImage.new("L", (width, height))
    m = mask.load()
    for y in range(height):
        v = int(255 * (y / max(1, height - 1)))
        for x in range(width):
            m[x, y] = v
    return PILImage.composite(top, base, mask)


def _cover_resize(img: PILImage.Image, box: tuple[int, int]) -> PILImage.Image:
    w, h = box
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
    left = max(0, (new_w - w) // 2)
    top = max(0, (new_h - h) // 2)
    return resized.crop((left, top, left + w, top + h))


def _openai_generate_image_png(prompt: str, model: str, size: str) -> bytes:
    api_key = _require_env("OPENAI_API_KEY")
    resp = httpx.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
        },
        timeout=120,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        raise RuntimeError(resp.text)
    data = resp.json()
    item = (data.get("data") or [{}])[0]

    png_bytes: bytes | None = None
    b64 = item.get("b64_json")
    if b64:
        png_bytes = base64.b64decode(b64)

    url = item.get("url")
    if (not png_bytes) and url:
        img = httpx.get(url, timeout=120)
        try:
            img.raise_for_status()
        except httpx.HTTPStatusError:
            raise RuntimeError(img.text)
        png_bytes = img.content

    if not png_bytes:
        raise RuntimeError("OpenAI returned no image")
    return png_bytes


def _openai_size_for_canvas(width: int, height: int) -> str:
    aspect = width / max(1, height)
    if aspect >= 1.18:
        return "1536x1024"
    if aspect <= 0.85:
        return "1024x1536"
    return "1024x1024"


def _openai_edit_image_png(image_png: bytes, prompt: str, model: str, size: str) -> bytes:
    api_key = _require_env("OPENAI_API_KEY")
    files = {"image": ("image.png", image_png, "image/png")}
    data = {"model": model, "prompt": prompt, "size": size}
    resp = httpx.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=180,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        raise RuntimeError(resp.text)
    payload = resp.json()
    item = (payload.get("data") or [{}])[0]
    b64 = item.get("b64_json")
    if isinstance(b64, str) and b64:
        return base64.b64decode(b64)
    url = item.get("url")
    if isinstance(url, str) and url:
        img = httpx.get(url, timeout=180)
        try:
            img.raise_for_status()
        except httpx.HTTPStatusError:
            raise RuntimeError(img.text)
        return img.content
    raise RuntimeError("OpenAI returned no edited image")


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = (text or "").strip().split()
    if not words:
        return []
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _fit_font_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_lines: int,
    start_size: int,
    min_size: int,
) -> tuple[ImageFont.ImageFont, list[str]]:
    size = start_size
    while size >= min_size:
        font = _load_font(size)
        lines = _wrap_text(draw, text, font, max_width=max_width)
        if not lines:
            return font, []
        if len(lines) <= max_lines:
            return font, lines
        size -= 2
    font = _load_font(min_size)
    return font, _wrap_text(draw, text, font, max_width=max_width)[:max_lines]


def _fit_contain(img: PILImage.Image, box: tuple[int, int]) -> PILImage.Image:
    w, h = box
    src_w, src_h = img.size
    scale = min(w / src_w, h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    return img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)


def _rounded_mask(size: tuple[int, int], radius: int) -> PILImage.Image:
    m = PILImage.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return m


def _supabase_insert(table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = _require_env("SUPABASE_URL").rstrip("/")
    key = _require_env("SUPABASE_SERVICE_ROLE_KEY")

    resp = httpx.post(
        f"{url}/rest/v1/{table}",
        headers={
            **_supabase_headers(key),
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=rows,
        timeout=120,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        raise RuntimeError(resp.text)
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Supabase returned unexpected response")
    return data


@mcp.tool()
def upload_screenshot(file_path: str, object_path: str | None = None) -> dict[str, Any]:
    p = Path(file_path).expanduser()
    if not p.exists():
        raise ValueError("file_path does not exist")

    img = PILImage.open(p).convert("RGBA")
    width, height = img.size

    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    final_object_path = object_path or f"screenshots/{uuid4().hex}.png"
    url = _supabase_upload_png(png_bytes, final_object_path)
    return {"url": url, "object_path": final_object_path, "width": width, "height": height}


@mcp.tool()
def generate_mockup_thumbnail(
    screenshot_urls: list[str],
    headline: str,
    subheadline: str | None = None,
    brand_hex: list[str] | None = None,
    device: str = "auto",
    size: str = "1200x627",
    background_mode: str = "ai",
    background_prompt: str | None = None,
    background_model: str | None = None,
    upload: bool = True,
    object_path: str | None = None,
) -> list[types.Content]:
    if not screenshot_urls:
        raise ValueError("screenshot_urls is required")

    width, height = _parse_size(size)
    bg_mode_used = "gradient"
    bg_model_used: str | None = None
    bg_prompt_used: str | None = None
    bg_fallback = False

    mode = (background_mode or "gradient").strip().lower()
    if mode == "ai":
        bg_model_used = background_model or _get_openai_bg_image_model()
        colors_str = ", ".join(brand_hex or [])
        bg_prompt_used = (
            background_prompt
            or f"""High-end abstract background for a SaaS product launch thumbnail.
Color palette: {colors_str or "dark navy, electric cyan, indigo glow"}.
Style: modern, premium, subtle gradient glow, soft depth, light noise texture, clean shapes, minimal.
No text, no logos, no UI, no watermarks, no people."""
        ).strip()
        try:
            bg_png = _openai_generate_image_png(bg_prompt_used, model=bg_model_used, size="1024x1024")
            bg_img = PILImage.open(BytesIO(bg_png)).convert("RGBA")
            bg = _cover_resize(bg_img, (width, height))
            bg_mode_used = "ai"
        except Exception:
            bg = _gradient_bg(width, height, brand_hex).convert("RGBA")
            bg_fallback = True
    else:
        bg = _gradient_bg(width, height, brand_hex).convert("RGBA")

    shots: list[PILImage.Image] = []
    for u in screenshot_urls[:3]:
        r = httpx.get(u, timeout=60)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError:
            raise RuntimeError(r.text)
        shots.append(PILImage.open(BytesIO(r.content)).convert("RGBA"))

    main = shots[0]
    aspect = main.size[0] / max(1, main.size[1])
    device_value = device
    if device_value == "auto":
        device_value = "laptop" if aspect >= 1.15 else "phone"

    canvas = bg
    draw = ImageDraw.Draw(canvas)

    pad = int(width * 0.06)
    text_w = int(width * 0.42)
    screen_w = width - pad * 2 - text_w
    screen_h = int(height * 0.72)
    screen_x = pad + text_w
    screen_y = int(height * 0.14)

    if device_value == "phone":
        screen_w = int(width * 0.34)
        screen_h = int(height * 0.76)
        screen_x = width - pad - screen_w
        screen_y = int(height * 0.12)

    frame_pad = int(min(screen_w, screen_h) * 0.06)
    bezel = int(frame_pad * 0.9)
    radius = int(min(screen_w, screen_h) * 0.06)

    frame_rect = (screen_x, screen_y, screen_x + screen_w, screen_y + screen_h)
    shadow = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle(frame_rect, radius=radius, fill=(0, 0, 0, 140))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    canvas.alpha_composite(shadow)

    draw.rounded_rectangle(frame_rect, radius=radius, fill=(17, 24, 39, 255))

    screen_rect = (
        screen_x + bezel,
        screen_y + bezel,
        screen_x + screen_w - bezel,
        screen_y + screen_h - bezel,
    )
    screen_inner_w = screen_rect[2] - screen_rect[0]
    screen_inner_h = screen_rect[3] - screen_rect[1]

    fitted = _fit_contain(main, (screen_inner_w, screen_inner_h))
    screen_bg = PILImage.new("RGBA", (screen_inner_w, screen_inner_h), (0, 0, 0, 0))
    ox = (screen_inner_w - fitted.size[0]) // 2
    oy = (screen_inner_h - fitted.size[1]) // 2
    screen_bg.alpha_composite(fitted, (ox, oy))

    mask = _rounded_mask((screen_inner_w, screen_inner_h), radius=max(8, radius - 8))
    canvas.paste(screen_bg, (screen_rect[0], screen_rect[1]), mask)

    if device_value == "laptop":
        base_h = int(screen_h * 0.13)
        base_rect = (
            screen_x + int(screen_w * 0.12),
            screen_y + screen_h + int(base_h * 0.15),
            screen_x + int(screen_w * 0.88),
            screen_y + screen_h + base_h,
        )
        draw.rounded_rectangle(base_rect, radius=int(base_h * 0.35), fill=(17, 24, 39, 210))

    hx = pad
    hy = int(height * 0.20)
    headline_box_w = text_w

    head_font, head_lines = _fit_font_to_box(
        draw,
        headline,
        max_width=headline_box_w,
        max_lines=3,
        start_size=int(height * 0.095),
        min_size=max(22, int(height * 0.055)),
    )
    sub_font, sub_lines = _fit_font_to_box(
        draw,
        subheadline or "",
        max_width=headline_box_w,
        max_lines=2,
        start_size=int(height * 0.048),
        min_size=max(18, int(height * 0.036)),
    )

    shadow = (0, 0, 0, 160)
    text_y = hy
    line_gap = int(head_font.size * 0.22) if hasattr(head_font, "size") else int(height * 0.02)
    for line in head_lines:
        draw.text((hx + 2, text_y + 2), line, fill=shadow, font=head_font)
        draw.text((hx, text_y), line, fill=(255, 255, 255, 255), font=head_font)
        bbox = draw.textbbox((0, 0), line, font=head_font)
        text_y += (bbox[3] - bbox[1]) + line_gap

    if sub_lines:
        text_y += int(height * 0.02)
        sub_gap = int(sub_font.size * 0.28) if hasattr(sub_font, "size") else int(height * 0.018)
        for line in sub_lines:
            draw.text((hx + 2, text_y + 2), line, fill=shadow, font=sub_font)
            draw.text((hx, text_y), line, fill=(226, 232, 240, 255), font=sub_font)
            bbox = draw.textbbox((0, 0), line, font=sub_font)
            text_y += (bbox[3] - bbox[1]) + sub_gap

    if len(shots) > 1:
        thumb_w = int(text_w * 0.45)
        thumb_h = int(thumb_w * 0.6)
        ty = int(height * 0.62)
        tx = pad
        for i, im in enumerate(shots[1:3]):
            tfit = _fit_contain(im, (thumb_w, thumb_h))
            tile = PILImage.new("RGBA", (thumb_w, thumb_h), (15, 23, 42, 180))
            tile.alpha_composite(
                tfit, ((thumb_w - tfit.size[0]) // 2, (thumb_h - tfit.size[1]) // 2)
            )
            tile_mask = _rounded_mask((thumb_w, thumb_h), radius=18)
            canvas.paste(tile, (tx, ty), tile_mask)
            tx += thumb_w + int(pad * 0.35)

    out = BytesIO()
    canvas.convert("RGBA").save(out, format="PNG")
    png_bytes = out.getvalue()

    public_url: str | None = None
    final_object_path: str | None = None
    if upload:
        final_object_path = object_path or f"thumbnails/{uuid4().hex}.png"
        public_url = _supabase_upload_png(png_bytes, final_object_path)

    meta = {
        "thumbnail_url": public_url,
        "object_path": final_object_path,
        "bucket": _supabase_bucket() if upload else None,
        "size": f"{width}x{height}",
        "device": device_value,
        "screenshots": screenshot_urls[:3],
        "background_mode": bg_mode_used,
        "background_model": bg_model_used,
        "background_prompt": bg_prompt_used,
        "background_fallback": bg_fallback,
    }
    return [
        types.TextContent(type="text", text=json.dumps(meta, ensure_ascii=False)),
        Image(data=png_bytes, format="png").to_image_content(),
    ]


@mcp.tool()
def generate_thumbnail(
    prompt: str,
    size: str = "1024x1024",
    model: str | None = None,
    upload: bool | None = None,
    object_path: str | None = None,
) -> list[types.Content]:
    image_model = model or _get_openai_image_model()
    png_bytes = _openai_generate_image_png(prompt, model=image_model, size=size)

    should_upload = upload
    if should_upload is None:
        should_upload = bool(_supabase_url() and _supabase_service_role_key())

    public_url: str | None = None
    final_object_path: str | None = None

    if should_upload:
        final_object_path = object_path or f"launches/{uuid4().hex}.png"
        public_url = _supabase_upload_png(png_bytes, final_object_path)

    meta = {
        "thumbnail_url": public_url,
        "object_path": final_object_path,
        "bucket": _supabase_bucket() if should_upload else None,
        "model": image_model,
        "size": size,
    }
    return [
        types.TextContent(type="text", text=json.dumps(meta, ensure_ascii=False)),
        Image(data=png_bytes, format="png").to_image_content(),
    ]


@mcp.tool()
def choose_platform(
    title: str,
    summary: str,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    model = _get_openai_text_model()
    platforms_list = platforms or ["linkedin", "twitter", "instagram"]
    prompt = f"""
You are a launch marketing strategist.

Decide which platforms are best for this product update and why.

Title: {title}
Summary: {summary}
Candidate platforms: {", ".join(platforms_list)}

Return JSON only with this schema:
{{
  "platforms": [
    {{
      "platform": "linkedin|twitter|instagram|...",
      "rationale": "string",
      "thumbnail_prompt": "string",
      "copy_guidelines": ["string", "..."]
    }}
  ]
}}
""".strip()

    text = _openai_responses(model=model, input_text=prompt)
    return _parse_json(text)


@mcp.tool()
def generate_copy(
    platform: str,
    title: str,
    summary: str,
    tone: str | None = None,
) -> dict[str, Any]:
    model = _get_openai_text_model()
    tone_value = tone or "clear, concise, confident"
    prompt = f"""
You are a specialist copywriter for {platform}.

Write a social media post for a product update.
Tone: {tone_value}

Title: {title}
Summary: {summary}

Return JSON only with this schema:
{{
  "platform": "{platform}",
  "copy": "string"
}}
""".strip()
    text = _openai_responses(model=model, input_text=prompt)
    return _parse_json(text)


@mcp.tool()
def save_draft(
    title: str,
    summary: str | None = None,
    repo_url: str | None = None,
    source_url: str | None = None,
    contents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not contents:
        raise ValueError("contents is required")

    launch_row = _supabase_insert(
        "launches",
        [
            {
                "title": title,
                "summary": summary,
                "repo_url": repo_url,
                "source_url": source_url,
                "status": "ready",
            }
        ],
    )[0]

    launch_id = launch_row.get("id")
    if not launch_id:
        raise RuntimeError("Supabase did not return launch id")

    platform_rows: list[dict[str, Any]] = []
    for c in contents:
        post_status = c.get("post_status") or c.get("status") or "draft"
        platform_rows.append(
            {
                "launch_id": launch_id,
                "platform": c.get("platform"),
                "copy": c.get("copy"),
                "thumbnail_url": c.get("thumbnail_url"),
                "video_url": c.get("video_url"),
                "video_status": c.get("video_status") or "none",
                "post_status": post_status,
            }
        )

    inserted = _supabase_insert("platform_contents", platform_rows)
    return {"launch": launch_row, "platform_contents": inserted}


@mcp.tool()
def launch_linkedin_from_screenshots(
    title: str,
    summary: str,
    screenshot_paths: list[str],
    brand_hex: list[str] | None = None,
    device: str = "laptop",
    size: str = "1200x627",
    background_mode: str = "ai",
) -> dict[str, Any]:
    if not screenshot_paths:
        raise ValueError("screenshot_paths is required")

    uploaded = [upload_screenshot(p) for p in screenshot_paths]
    screenshot_urls = [u["url"] for u in uploaded]

    model = _get_openai_text_model()
    prompt = f"""
You are a senior launch copywriter + designer for LinkedIn.
Voice: indie hacker (builder vibe), crisp, product-led, minimal fluff.

Write:
1) A short headline (max 6 words) that names the outcome (no "Memperkenalkan/Introducing").
2) A short subheadline (max 7 words) that clarifies the value.
3) A LinkedIn post copy (600-900 chars) in Indonesian with:
   - 1 hook line (builder vibe)
   - 3 bullets (feature -> benefit), very short
   - 1 short CTA without any link
   - max 2 emojis total (optional)
   - up to 3 hashtags at the very end
4) A thumbnail_edit_prompt: instructions for an image model to create a premium SaaS marketing thumbnail by remixing the provided screenshot.
   - Must keep the screenshot recognizable/readable
   - No random text/logos/watermarks/people
   - Leave clean negative space on the left for headline/subheadline overlay

Title: {title}
Summary: {summary}
Brand colors (hex): {", ".join(brand_hex or [])}

Return JSON only with this schema:
{{
  "headline": "string",
  "subheadline": "string",
  "copy": "string",
  "thumbnail_edit_prompt": "string"
}}
""".strip()

    text = _openai_responses(model=model, input_text=prompt)
    plan = _parse_json(text)

    width, height = _parse_size(size)
    thumbnail_url: str | None = None
    try:
        p0 = Path(screenshot_paths[0]).expanduser()
        shot = PILImage.open(p0).convert("RGBA")
        buf = BytesIO()
        shot.save(buf, format="PNG")
        shot_png = buf.getvalue()

        edit_model = _get_openai_bg_image_model()
        edit_size = _openai_size_for_canvas(width, height)
        colors_str = ", ".join(brand_hex or [])
        edit_prompt = (
            str(plan.get("thumbnail_edit_prompt") or "").strip()
            or f"""Create a premium SaaS marketing thumbnail by remixing the provided screenshot.
Color palette: {colors_str or "teal and purple"}.
Style: modern, high-contrast, clean, subtle glow, soft depth, minimal.
Constraints: keep the screenshot recognizable and readable; no new logos; no watermarks; no random UI text; no people; leave clear negative space on the left for headline text overlay."""
        )
        ai_png = _openai_edit_image_png(shot_png, prompt=edit_prompt, model=edit_model, size=edit_size)
        ai_img = PILImage.open(BytesIO(ai_png)).convert("RGBA")
        canvas = _cover_resize(ai_img, (width, height))

        draw = ImageDraw.Draw(canvas)
        pad = int(width * 0.06)
        text_w = int(width * 0.46)
        hx = pad
        hy = int(height * 0.18)

        head_font, head_lines = _fit_font_to_box(
            draw,
            str(plan.get("headline") or title),
            max_width=text_w,
            max_lines=3,
            start_size=int(height * 0.10),
            min_size=max(22, int(height * 0.055)),
        )
        sub_font, sub_lines = _fit_font_to_box(
            draw,
            str(plan.get("subheadline") or ""),
            max_width=text_w,
            max_lines=2,
            start_size=int(height * 0.05),
            min_size=max(18, int(height * 0.036)),
        )

        shadow = (0, 0, 0, 170)
        text_y = hy
        line_gap = int(getattr(head_font, "size", int(height * 0.08)) * 0.22)
        for line in head_lines:
            draw.text((hx + 3, text_y + 3), line, fill=shadow, font=head_font)
            draw.text((hx, text_y), line, fill=(255, 255, 255, 255), font=head_font)
            bbox = draw.textbbox((0, 0), line, font=head_font)
            text_y += (bbox[3] - bbox[1]) + line_gap

        if sub_lines:
            text_y += int(height * 0.02)
            sub_gap = int(getattr(sub_font, "size", int(height * 0.04)) * 0.28)
            for line in sub_lines:
                draw.text((hx + 2, text_y + 2), line, fill=shadow, font=sub_font)
                draw.text((hx, text_y), line, fill=(226, 232, 240, 255), font=sub_font)
                bbox = draw.textbbox((0, 0), line, font=sub_font)
                text_y += (bbox[3] - bbox[1]) + sub_gap

        out = BytesIO()
        canvas.save(out, format="PNG")
        final_png = out.getvalue()
        thumbnail_url = _supabase_upload_png(final_png, f"thumbnails/{uuid4().hex}.png")
    except Exception:
        mock = generate_mockup_thumbnail(
            screenshot_urls=screenshot_urls,
            headline=str(plan.get("headline") or title),
            subheadline=str(plan.get("subheadline") or ""),
            brand_hex=brand_hex,
            device=device,
            size=size,
            background_mode=background_mode,
            background_prompt=None,
            upload=True,
        )
        meta = _parse_json(mock[0].text)
        thumbnail_url = meta.get("thumbnail_url")

    saved = save_draft(
        title=title,
        summary=summary,
        contents=[
            {
                "platform": "linkedin",
                "copy": str(plan.get("copy") or ""),
                "thumbnail_url": thumbnail_url,
                "post_status": "draft",
            }
        ],
    )

    return {
        "launch_id": saved.get("launch", {}).get("id"),
        "thumbnail_url": thumbnail_url,
        "screenshot_urls": screenshot_urls,
        "saved": saved,
    }


if __name__ == "__main__":
    mcp.run()
