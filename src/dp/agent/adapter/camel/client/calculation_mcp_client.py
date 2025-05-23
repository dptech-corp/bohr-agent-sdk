import functools
from typing import Callable, Optional

from camel.toolkits.mcp_toolkit import MCPClient
from mcp import Tool


class CalculationMCPClient(MCPClient):
    def __init__(
        self,
        *args,
        executor: Optional[str] = None,
        storage: Optional[str] = None,
        **kwargs,
    ):
        """Calculation MCP client
        extended from camel.toolkits.mcp_toolkit.MCPClient

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
        super().__init__(*args, **kwargs)
        self.executor = executor
        self.storage = storage

    def _merge_default_args(self, kwargs: dict) -> dict:
        if "executor" not in kwargs:
            kwargs["executor"] = self.executor
        if "storage" not in kwargs:
            kwargs["storage"] = self.storage
        return kwargs

    def generate_function_from_mcp_tool(self, mcp_tool: Tool) -> Callable:
        base_fn: Callable = super().generate_function_from_mcp_tool(mcp_tool)

        @functools.wraps(base_fn)
        async def wrapper(**kwargs):
            kwargs = self._merge_default_args(kwargs)
            return await base_fn(**kwargs)

        wrapper.__signature__ = base_fn.__signature__
        wrapper.__doc__ = base_fn.__doc__
        wrapper.__annotations__ = base_fn.__annotations__
        wrapper.__name__ = base_fn.__name__
        return wrapper
