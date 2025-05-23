
import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import StdioServerParameters
from dp.agent.adapter.adk import CalculationMCPToolset


async def test_mcp_tool():
    with open("fake_data", "w") as f:
        f.write("This is fake training data.")

    exit_stack = AsyncExitStack()
    try:
        server_script_path = str(Path(__file__).parent.parent.parent
                                 / "examples" / "calculation" / "server.py")
        tools = await CalculationMCPToolset(
            connection_params=StdioServerParameters(
                command='python3',
                args=[server_script_path],
            ),
            executor={
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
            },
            storage={
                "type": "bohrium",
                "username": os.environ.get("BOHRIUM_USERNAME"),
                "password": os.environ.get("BOHRIUM_PASSWORD"),
                "project_id": int(os.environ.get("BOHRIUM_PROJECT_ID")),
            },
        ).get_tools()
        tools = {tool.name: tool for tool in tools}
        tool = tools["run_dp_train"]
        result = await tool.run_async(args={
            "training_data": os.path.abspath("fake_data"),
        }, tool_context=None)
        results = json.loads(result.content[0].text)
        assert isinstance(results["model"], str)
        assert isinstance(results["log"], str)
        assert isinstance(results["lcurve"], str)
    finally:
        await exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(test_mcp_tool())
