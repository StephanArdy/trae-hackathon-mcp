import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    python = repo_dir / ".venv" / "bin" / "python"
    server_script = repo_dir / "main.py"

    screenshot = repo_dir.parents[0] / "quote-card" / "images" / "app" / "image.png"

    server_params = StdioServerParameters(command=str(python), args=[str(server_script)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "launch_linkedin_from_screenshots",
                {
                    "title": "QuoteCard Studio — screenshot-ready quote cards",
                    "summary": "Pilih quote → atur ratio (1:1, 16:9, 9:16), theme/accent/background/alignment → buka Render mode (URL render-only) untuk hasil pixel yang selalu sama. Cocok untuk screenshot automation dan template sosial media yang repeatable.",
                    "screenshot_paths": [str(screenshot)],
                    "brand_hex": ["#2DD4BF", "#A78BFA"],
                    "device": "laptop",
                    "size": "1200x627",
                },
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
