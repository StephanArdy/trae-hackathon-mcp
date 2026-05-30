## LaunchPilot MCP (Python)

### Python version

MCP Python SDK butuh Python `>=3.11`.

### Env

Copy `.env.example` to `.env` and fill:

- `OPENAI_API_KEY`
- `OPENAI_TEXT_MODEL` (default: `gpt-5.5`)
- `OPENAI_IMAGE_MODEL` (default: `gpt-image-1`, demo: `gpt-image-2`)
- `SUPABASE_URL` (optional; required for upload & save_draft)
- `SUPABASE_SERVICE_ROLE_KEY` (optional; required for upload & save_draft)
- `SUPABASE_STORAGE_BUCKET` (default: `image`)

`main.py` bakal auto-load `.env` yang berada di folder yang sama dengan file `main.py`, jadi aman walaupun working directory beda.

### Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Run (stdio)

```bash
python main.py
```

### Tools

- `generate_thumbnail(prompt, size?, model?, upload?, object_path?)`
- `upload_screenshot(file_path, object_path?)`
- `generate_mockup_thumbnail(screenshot_urls, headline, subheadline?, brand_hex?, device?, size?, upload?, object_path?)`
- `launch_linkedin_from_screenshots(title, summary, screenshot_paths, brand_hex?, device?, size?)`
- `choose_platform(title, summary, platforms?)`
- `generate_copy(platform, title, summary, tone?)`
- `save_draft(title, summary?, repo_url?, source_url?, contents)`

### Trae MCP config (local)

```json
{
  "mcpServers": {
    "launchpilot": {
      "command": "/Users/saputra/Documents/learnings/trae/trae-hackathon-mcp/.venv/bin/python",
      "args": ["/Users/saputra/Documents/learnings/trae/trae-hackathon-mcp/main.py"],
      "env": {}
    }
  }
}
```

Kalau mau override tanpa `.env`, isi `env` di atas (mis. `OPENAI_IMAGE_MODEL=gpt-image-2` pas demo).
