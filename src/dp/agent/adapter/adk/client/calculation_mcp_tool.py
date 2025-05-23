from typing import List, Optional

from google.adk.tools.mcp_tool import MCPTool, MCPToolset


class CalculationMCPTool(MCPTool):
    def __init__(
        self,
        executor: Optional[dict] = None,
        storage: Optional[dict] = None,
    ):
        """Calculation MCP tool
        extended from google.adk.tools.mcp_tool.MCPTool

        Args:
            executor: The executor configuration of the calculation tool.
                It is a dict where the "type" field specifies the executor
                type, and other fields are the keyword arguments of the
                corresponding executor type.
            storage: The storage configuration for storing artifacts. It is
                a dict where the "type" field specifies the storage type,
                and other fields are the keyword arguments of the
                corresponding storage type.
        """
        self.executor = executor
        self.storage = storage

    async def run_async(self, args, **kwargs):
        if "executor" not in args:
            args["executor"] = self.executor
        if "storage" not in args:
            args["storage"] = self.storage
        return await super().run_async(args=args, **kwargs)


class CalculationMCPToolset(MCPToolset):
    def __init__(
        self,
        executor: Optional[dict] = None,
        storage: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.executor = executor
        self.storage = storage

    async def get_tools(self, *args, **kwargs) -> List[CalculationMCPTool]:
        tools = await super().get_tools(*args, **kwargs)
        calc_tools = []
        for tool in tools:
            calc_tool = CalculationMCPTool(
                executor=self.executor, storage=self.storage)
            calc_tool.__dict__.update(tool.__dict__)
            calc_tools.append(calc_tool)
        return calc_tools
