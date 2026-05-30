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
                "generate_thumbnail",
                {"prompt": "Minimal flat app icon of a rocket on dark background", "size": "1024x1024"},
            )
            if result.isError:
                print(result.content[0].text)
                return
            meta = result.content[0]
            img = result.content[1]
            print(meta.text)
            print({"type": img.type, "mimeType": img.mimeType, "data_len": len(img.data)})


if __name__ == "__main__":
    asyncio.run(main())
