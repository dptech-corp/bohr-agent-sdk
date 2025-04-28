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
        func_name = mcp_tool.name
        func_desc = mcp_tool.description or "No description provided."
        parameters_schema = mcp_tool.inputSchema.get("properties", {})
        required_params = mcp_tool.inputSchema.get("required", [])

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        annotations = {}
        defaults: Dict[str, Any] = {}

        for param_name, param_schema in parameters_schema.items():
            param_type = param_schema.get("type", "Any")
            param_type = type_map.get(param_type, Any)
            annotations[param_name] = param_type
            if param_name not in required_params:
                defaults[param_name] = None

        async def dynamic_function(**kwargs):
            from mcp.types import CallToolResult

            kwargs = self._merge_default_args(kwargs)

            missing_params: Set[str] = set(required_params) - set(kwargs.keys())
            if missing_params:
                logger.warning(f"Missing required parameters: {missing_params}")
                return "Missing required parameters."

            if not self._session:
                logger.error("MCP Client is not connected. Call `connection()` first.")
                raise RuntimeError("MCP Client is not connected. Call `connection()` first.")

            try:
                result: CallToolResult = await self._session.call_tool(func_name, kwargs)
            except Exception as e:
                logger.error(f"Failed to call MCP tool '{func_name}': {e!s}")
                raise e

            if not result.content or len(result.content) == 0:
                return "No data available for this request."

            try:
                content = result.content[0]
                if content.type == "text":
                    return content.text
                elif content.type == "image":
                    if hasattr(content, "url") and content.url:
                        return f"Image available at: {content.url}"
                    return "Image content received (data URI not shown)"
                elif content.type == "embedded_resource":
                    if hasattr(content, "name") and content.name:
                        return f"Embedded resource: {content.name}"
                    return "Embedded resource received"
                else:
                    return f"Received content of type '{content.type}' which is not fully supported yet."
            except (IndexError, AttributeError) as e:
                logger.error(f"Error processing content from MCP tool response: {e!s}")
                raise e

        dynamic_function.__name__ = func_name
        dynamic_function.__doc__ = func_desc
        dynamic_function.__annotations__ = annotations

        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name=param,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=(None if param in ("executor", "storage") else defaults.get(param, inspect.Parameter.empty)),
                    annotation=annotations[param],
                )
                for param in parameters_schema.keys()
            ]
        )
        dynamic_function.__signature__ = sig

        return dynamic_function