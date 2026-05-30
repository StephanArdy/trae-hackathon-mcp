import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


async def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    python = repo_dir / ".venv" / "bin" / "python"
    server_script = repo_dir / "main.py"

    server_params = StdioServerParameters(command=str(python), args=[str(server_script)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            thumb = await session.call_tool(
                "generate_thumbnail",
                {
                    "prompt": "Clean SaaS product thumbnail: feature launch workflow cards, modern UI, blue accents",
                    "size": "1024x1024",
                    "upload": True,
                },
            )
            if thumb.isError:
                print(thumb.content[0].text)
                return
            meta_text = next(
                (c.text for c in thumb.content if getattr(c, "type", None) == "text" and getattr(c, "text", None)),
                "{}",
            )
            meta = _parse_json(meta_text)
            thumbnail_url = meta.get("thumbnail_url")

            saved = await session.call_tool(
                "save_draft",
                {
                    "title": "LaunchPilot: Draft Launch Posts",
                    "summary": "Generate thumbnails + copywriting drafts for a shipped feature and store it for review.",
                    "contents": [
                        {
                            "platform": "linkedin",
                            "copy": "Shipping is easy. Launching is the hard part. LaunchPilot drafts your launch assets in minutes.",
                            "thumbnail_url": thumbnail_url,
                        }
                    ],
                },
            )
            print(saved.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
