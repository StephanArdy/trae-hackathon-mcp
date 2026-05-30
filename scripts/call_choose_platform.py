import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    python = repo_dir / ".venv" / "bin" / "python"
    server_script = repo_dir / "main.py"

    server_params = StdioServerParameters(command=str(python), args=[str(server_script)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "choose_platform",
                {
                    "title": "LaunchPilot: Draft Launch Posts",
                    "summary": "Generate thumbnails + copywriting drafts for a shipped feature and store it for review.",
                    "platforms": ["linkedin", "twitter", "instagram"],
                },
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
