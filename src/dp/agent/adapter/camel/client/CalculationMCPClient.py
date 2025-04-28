import inspect
import json
import os
import shlex
from contextlib import AsyncExitStack, asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    cast,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from mcp import ClientSession, ListToolsResult, Tool

from camel.logger import get_logger
from camel.toolkits.mcp_toolkit import MCPClient
import functools

logger = get_logger(__name__)

class CalculationMCPClient(MCPClient):
    def __init__(
        self,
        command_or_url: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        default_executor: Optional[str] = None,
        default_storage: Optional[str] = None,
    ):
        super().__init__(command_or_url, args, env, timeout, headers)
        self._default_executor = default_executor
        self._default_storage = default_storage

    def _merge_default_args(self, kwargs: dict) -> dict:
        if "executor" not in kwargs and self._default_executor is not None:
            kwargs["executor"] = self._default_executor
        if "storage" not in kwargs and self._default_storage is not None:
            kwargs["storage"] = self._default_storage
        return kwargs

    def generate_function_from_mcp_tool(self, mcp_tool: "Tool") -> Callable:
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