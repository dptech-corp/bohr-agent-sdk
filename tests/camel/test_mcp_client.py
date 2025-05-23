import asyncio
import json
import os
from pathlib import Path

from dp.agent.adapter.camel import CalculationMCPClient
from camel.toolkits.mcp_toolkit import MCPToolkit


async def test_mcp_client():
    with open("fake_data", "w") as f:
        f.write("This is fake training data.")

    server_script_path = str(Path(__file__).parent.parent.parent
                             / "examples" / "calculation" / "server.py")
    mcp_toolkit = MCPToolkit(servers=[CalculationMCPClient(
        command_or_url='python3',
        args=[server_script_path],
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
    )])
    try:
        await mcp_toolkit.connect()

        tools = mcp_toolkit.get_tools()
        tools = {tool.func.__name__: tool for tool in tools}
        tool = tools["run_dp_train"]
        result = await tool(**{
            "training_data": os.path.abspath("fake_data"),
        })
        results = json.loads(result)
        assert isinstance(results["model"], str)
        assert isinstance(results["log"], str)
        assert isinstance(results["lcurve"], str)
    finally:
        await mcp_toolkit.disconnect()

if __name__ == "__main__":
    asyncio.run(test_mcp_client())
