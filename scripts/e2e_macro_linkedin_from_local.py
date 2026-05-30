import asyncio
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image as PILImage


async def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    python = repo_dir / ".venv" / "bin" / "python"
    server_script = repo_dir / "main.py"

    tmp = Path(tempfile.gettempdir()) / "launchpilot-shot-macro.png"
    img = PILImage.new("RGB", (1400, 900), (17, 24, 39))
    img.save(tmp, format="PNG")

    server_params = StdioServerParameters(command=str(python), args=[str(server_script)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "launch_linkedin_from_screenshots",
                {
                    "title": "LaunchPilot: Draft Launch Posts",
                    "summary": "Turn shipped features into launch-ready copy + thumbnails and save drafts for review.",
                    "screenshot_paths": [str(tmp)],
                    "brand_hex": ["#0EA5E9", "#6366F1"],
                    "device": "laptop",
                    "size": "1200x627",
                },
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
