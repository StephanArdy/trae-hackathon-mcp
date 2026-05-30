import asyncio
import json
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image as PILImage


async def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    python = repo_dir / ".venv" / "bin" / "python"
    server_script = repo_dir / "main.py"

    tmp = Path(tempfile.gettempdir()) / "launchpilot-shot.png"
    img = PILImage.new("RGB", (1400, 900), (30, 41, 59))
    img.save(tmp, format="PNG")

    server_params = StdioServerParameters(command=str(python), args=[str(server_script)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            uploaded = await session.call_tool("upload_screenshot", {"file_path": str(tmp)})
            info = json.loads(uploaded.content[0].text)

            mock = await session.call_tool(
                "generate_mockup_thumbnail",
                {
                    "screenshot_urls": [info["url"]],
                    "headline": "Draft launch posts in minutes",
                    "subheadline": "Ship → Generate → Review",
                    "brand_hex": ["#0EA5E9", "#6366F1"],
                    "size": "1200x627",
                    "device": "laptop",
                    "upload": True,
                },
            )
            print(mock.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
