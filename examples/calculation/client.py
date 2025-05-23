import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str, env: dict = None):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def call_tool(self, tool_name, arguments):
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    executor = {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": os.environ.get("BOHRIUM_USERNAME"),
                "password": os.environ.get("BOHRIUM_PASSWORD"),
                "program_id": int(os.environ.get("BOHRIUM_PROJECT_ID")),
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/ubuntu:22.04-py3.10",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c2_m4_cpu",
                },
            },
        },
    }
    storage = {
        "type": "bohrium",
        "username": os.environ.get("BOHRIUM_USERNAME"),
        "password": os.environ.get("BOHRIUM_PASSWORD"),
        "project_id": int(os.environ.get("BOHRIUM_PROJECT_ID")),
    }
    client = MCPClient()
    try:
        await client.connect_to_server(
            "server.py",
        )
        result = await client.call_tool(
            "run_dp_train",
            arguments={
                "training_data": "bohrium://13756/27666/store/upload/training_data.tgz",
                "storage": storage,
                "executor": executor,
            },
        )
        results = result.content[0].text
        print("results", results)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())