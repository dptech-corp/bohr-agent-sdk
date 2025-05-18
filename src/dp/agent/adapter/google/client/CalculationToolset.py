# my_mcp_toolset.py
from functools import wraps
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack
import sys
from types import TracebackType
from typing import List, Optional, TextIO, Tuple, Type
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, MCPTool
from pydantic import BaseModel
try:
  from mcp import StdioServerParameters

except ImportError as e:
  import sys

class SseServerParams(BaseModel):
  """Parameters for the MCP SSE connection.

  See MCP SSE Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/sse.py
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5
  sse_read_timeout: float = 60 * 5

class _DefaultArgsMCPTool:


    def __init__(
        self,
        inner: MCPTool,
        default_executor: Optional[str],
        default_storage: Optional[str],
    ):
        self._inner = inner
        self._default_executor = default_executor
        self._default_storage = default_storage
        self.args_schema = getattr(inner, "args_schema", None)
        

    async def run_async(self, args: dict, tool_context=None):
        if "executor" not in args and self._default_executor is not None:
            args["executor"] = self._default_executor
        if "storage" not in args and self._default_storage is not None:
            args["storage"] = self._default_storage
        return await self._inner.run_async(args=args, tool_context=tool_context)

    async def __call__(self, **kwargs):
        return await self.run_async(args=kwargs)
    
    def __getattr__(self, item):
        return getattr(self._inner, item)


class CalculationToolset(MCPToolset):

    def __init__(
        self,
        *,
        connection_params,
        default_executor: Optional[str] = None,
        default_storage: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(connection_params=connection_params, **kwargs)
        self._default_executor = default_executor
        self._default_storage = default_storage

    async def load_tools(self) -> List[MCPTool]:
        raw_tools: List[MCPTool] = await super().load_tools()
        return [
            _DefaultArgsMCPTool(t, self._default_executor, self._default_storage)
            for t in raw_tools
        ]
    @classmethod
    async def from_server(
        cls,
        *,
        connection_params: StdioServerParameters | SseServerParams,
        default_executor: Optional[str] = None,      
        default_storage: Optional[str] = None,     
        async_exit_stack: Optional[AsyncExitStack] = None,
        errlog: TextIO = sys.stderr,
    ) -> Tuple[List[MCPTool], AsyncExitStack]:
        async_exit_stack = async_exit_stack or AsyncExitStack()

        toolset = cls(
            connection_params=connection_params,
            default_executor=default_executor,  
            default_storage=default_storage,    
            exit_stack=async_exit_stack,
            errlog=errlog,
        )

        await async_exit_stack.enter_async_context(toolset)
        tools = await toolset.load_tools()
        return tools, async_exit_stack