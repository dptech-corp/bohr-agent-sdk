import asyncio
import json
import jsonpickle
import logging
from copy import deepcopy
from typing import Callable, List, Optional, Any

from mcp import ClientSession, types
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool import MCPTool, MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager

from ..utils import get_logger
logger = get_logger(__name__)


async def logging_handler(
    params: types.LoggingMessageNotificationParams,
    tool_context: ToolContext = None,
) -> None:
    logger.log(getattr(logging, params.level.upper()), params.data)


class MCPSessionManagerWithLoggingCallback(MCPSessionManager):
    def __init__(
      self,
      logging_callback=None,
      **kwargs,
    ):
        super().__init__(**kwargs)
        self.logging_callback = logging_callback

    async def create_session(self, *args, **kwargs) -> ClientSession:
        session = await super().create_session(*args, **kwargs)
        session._logging_callback = self.logging_callback
        return session


class CalculationMCPTool(MCPTool):
    def __init__(
        self,
        executor: Optional[dict] = None,
        storage: Optional[dict] = None,
        async_mode: bool = False,
        wait: bool = True,
        submit_tool: Optional[MCPTool] = None,
        query_tool: Optional[MCPTool] = None,
        terminate_tool: Optional[MCPTool] = None,
        results_tool: Optional[MCPTool] = None,
        query_interval: int = 10,
        logging_callback: Callable = logging_handler,
        override: bool = True,
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
            async_mode: Submit and query until the job finishes, instead of
                waiting in single connection
            wait: Wait for the job to finish or directly return
            submit_tool: The tool of submitting job
            query_tool: The tool of querying job status
            terminate_tool: The tool of terminating job
            results_tool: The tool of getting job results
            query_interval: Time interval of querying job status
            logging_callback: Callback function for server notifications
            override: Override storage and executor in tool params or not
        """
        self.executor = executor
        self.storage = storage
        self.async_mode = async_mode
        self.submit_tool = submit_tool
        self.query_tool = query_tool
        self.terminate_tool = terminate_tool
        self.results_tool = results_tool
        self.query_interval = query_interval
        self.wait = wait
        self.logging_callback = logging_callback
        self.override = override

    async def log(self, level: str, message: Any, tool_context: ToolContext):
        await self.logging_callback(types.LoggingMessageNotificationParams(
            data=message, level=level.lower()), tool_context=tool_context)

    async def run_async(self, args, tool_context: ToolContext, **kwargs):
        # TODO: add progress callback when run_async
        args = deepcopy(args)
        if self.override:
            args["executor"] = self.executor
            args["storage"] = self.storage
        if not args.get("executor"):
            args.pop("executor", None)
        if not args.get("storage"):
            args.pop("storage", None)
        if not self.async_mode and self.wait:
            return await super().run_async(
                args=args, tool_context=tool_context, **kwargs)

        executor = args.get("executor")
        storage = args.get("storage")
        res = await self.submit_tool.run_async(
            args=args, tool_context=tool_context, **kwargs)
        if isinstance(res, dict):
            res = types.CallToolResult.model_validate(res)
        if res.isError:
            logger.error(res.content[0].text)
            return res
        job_id = json.loads(res.content[0].text)["job_id"]
        job_info = res.content[0].job_info
        await self.log("info", "Job submitted (ID: %s)" % job_id, tool_context)
        if job_info.get("extra_info"):
            await self.log("info", job_info["extra_info"], tool_context)
        if not self.wait:
            res.content[0].text = json.dumps({
                "job_id": job_id,
                "status": "Running",
                "extra_info": job_info.get("extra_info"),
            })
            return res

        while True:
            res = await self.query_tool.run_async(
                args={"job_id": job_id, "executor": executor},
                tool_context=tool_context, **kwargs)
            if isinstance(res, dict):
                res = types.CallToolResult.model_validate(res)
            if res.isError:
                logger.error(res.content[0].text)
            else:
                status = res.content[0].text
                await self.log("info", "Job %s status is %s" % (
                    job_id, status), tool_context)
                if status != "Running":
                    break
            await asyncio.sleep(self.query_interval)

        res = await self.results_tool.run_async(
            args={"job_id": job_id, "executor": executor, "storage": storage},
            tool_context=tool_context, **kwargs)
        if isinstance(res, dict):
            res = types.CallToolResult.model_validate(res)
        if res.isError:
            await self.log("error", "Job %s failed: %s" % (
                job_id, res.content[0].text), tool_context)
        else:
            await self.log("info", "Job %s result is %s" % (
                job_id, jsonpickle.loads(res.content[0].text)), tool_context)
        res.content[0].job_info = {**job_info,
                                   **getattr(res.content[0], "job_info", {})}
        return res


class CalculationMCPToolset(MCPToolset):
    def __init__(
        self,
        executor: Optional[dict] = None,
        storage: Optional[dict] = None,
        executor_map: Optional[dict] = None,
        async_mode: bool = False,
        wait: bool = True,
        logging_callback: Callable = logging_handler,
        override: bool = True,
        **kwargs,
    ):
        """
        Calculation MCP toolset

        Args:
            executor: The default executor configuration of the calculation
                tools. It is a dict where the "type" field specifies the
                executor type, and other fields are the keyword arguments of
                the corresponding executor type.
            storage: The storage configuration for storing artifacts. It is
                a dict where the "type" field specifies the storage type,
                and other fields are the keyword arguments of the
                corresponding storage type.
            executor_map: A dict mapping from tool name to executor
                configuration for specifying particular executor for certain
                tools
            async_mode: Submit and query until the job finishes, instead of
                waiting in single connection
            wait: Wait for the job to finish or directly return
            logging_callback: Callback function for server notifications
            override: Override storage and executor in tool params or not
        """
        super().__init__(**kwargs)
        self.logging_callback = logging_callback
        self._mcp_session_manager = MCPSessionManagerWithLoggingCallback(
            connection_params=self._connection_params,
            errlog=self._errlog,
            logging_callback=logging_callback,
        )
        self.executor = executor
        self.storage = storage
        self.wait = wait
        self.executor_map = executor_map or {}
        self.async_mode = async_mode
        self.query_tool = None
        self.terminate_tool = None
        self.results_tool = None
        self.override = override

    async def get_tools(self, *args, **kwargs) -> List[CalculationMCPTool]:
        tools = await super().get_tools(*args, **kwargs)
        tools = {tool.name: tool for tool in tools}
        self.query_tool = tools.get("query_job_status")
        self.terminate_tool = tools.get("terminate_job")
        self.results_tool = tools.get("get_job_results")
        calc_tools = []
        for tool in tools.values():
            if tool.name.startswith("submit_") or tool.name in [
                    "query_job_status", "terminate_job", "get_job_results"]:
                continue
            calc_tool = CalculationMCPTool(
                executor=self.executor_map.get(tool.name, self.executor),
                storage=self.storage,
                async_mode=self.async_mode,
                wait=self.wait,
                submit_tool=tools.get("submit_" + tool.name),
                query_tool=tools.get("query_job_status"),
                terminate_tool=tools.get("terminate_job"),
                results_tool=tools.get("get_job_results"),
                logging_callback=self.logging_callback,
                override=self.override,
            )
            calc_tool.__dict__.update(tool.__dict__)
            calc_tool.is_long_running = not self.wait
            calc_tools.append(calc_tool)
        return calc_tools


class BackgroundJobWatcher:
    def __init__(self, toolset: "CalculationMCPToolset"):
        self.long_running_ids = []
        self.long_running_jobs = {}
        self.status = {}
        self.toolset = toolset

    def record_event(self, event):
        if event.long_running_tool_ids:
            self.long_running_ids += event.long_running_tool_ids

        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part
                    and part.function_response
                    and part.function_response.id in self.long_running_ids
                    and "result" in part.function_response.response
                ):
                    result = part.function_response.response["result"]
                    if isinstance(result, dict):
                        result = types.CallToolResult.model_validate(result)
                    if not result.isError:
                        results = json.loads(result.content[0].text)
                        job_id = results["job_id"]
                        self.long_running_jobs[job_id] = part.function_response
                        self.status[job_id] = "Running"

    async def watch_jobs(self):
        for job_id in self.status.keys():
            if self.status[job_id] != "Running":
                continue
            res = await self.toolset.query_tool.run_async(
                args={"job_id": job_id, "executor": self.toolset.executor},
                tool_context=None)
            if isinstance(res, dict):
                res = types.CallToolResult.model_validate(res)
            if res.isError:
                logger.error(res.content[0].text)
                continue
            status = res.content[0].text
            if status != "Running":
                res = await self.toolset.results_tool.run_async(
                    args={"job_id": job_id, "executor": self.toolset.executor,
                          "storage": self.toolset.storage},
                    tool_context=None)
                if isinstance(res, dict):
                    res = types.CallToolResult.model_validate(res)
                job_info = getattr(res.content[0], "job_info", {})
                response = self.long_running_jobs[job_id]
                if res.isError:
                    err_msg = res.content[0].text
                    if err_msg.startswith("Error executing tool"):
                        err_msg = err_msg[err_msg.find(":")+2:]
                    result = f"Error executing tool {response.name}: {err_msg}"
                else:
                    result = res.content[0].text
                res = response.response["result"]
                job_info.update(getattr(res.content[0], "job_info", {}))
                res.content[0].job_info = job_info
                res.content[0].text = result
            self.status[job_id] = status
            yield job_id, status

    def get_response(self, job_id):
        return self.long_running_jobs[job_id]

    def get_status(self, job_id):
        return self.status[job_id]
