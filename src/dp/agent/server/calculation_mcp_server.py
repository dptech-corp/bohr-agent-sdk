import inspect
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Annotated, Literal, Optional, List, Dict

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.context_injection import (
    find_context_parameter,
)
from mcp.server.fastmcp.utilities.func_metadata import (
    ArgModelBase,
    func_metadata,
)
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel, Field, create_model
from starlette.responses import JSONResponse
from starlette.routing import Route

from .executor import executor_dict
from .storage import storage_dict
from .utils import get_logger, JobResult, Tool
logger = get_logger(__name__)
CALCULATION_MCP_WORKDIR = os.getenv("CALCULATION_MCP_WORKDIR", os.getcwd())


def parse_uri(uri):
    scheme = urlparse(uri).scheme
    if scheme == "":
        key = uri
        scheme = "local"
    else:
        key = uri[len(scheme)+3:]
    return scheme, key


def init_storage(storage_config: Optional[dict] = None):
    if not storage_config:
        storage_config = {"type": "local"}
    storage_config = deepcopy(storage_config)
    storage_type = storage_config.pop("type")
    storage = storage_dict[storage_type](**storage_config)
    return storage_type, storage


def init_executor(executor_config: Optional[dict] = None):
    if not executor_config:
        executor_config = {"type": "local"}
    executor_config = deepcopy(executor_config)
    executor_type = executor_config.pop("type")
    return executor_type, executor_dict[executor_type](**executor_config)


@contextmanager
def set_directory(workdir: str):
    cwd = os.getcwd()
    workdir = os.path.join(CALCULATION_MCP_WORKDIR, workdir)
    os.makedirs(workdir, exist_ok=True)
    try:
        os.chdir(workdir)
        yield
    finally:
        os.chdir(cwd)


def load_job_info():
    with open("job.json", "r") as f:
        return json.load(f)


def query_job_status(job_id: str, executor: Optional[dict] = None
                     ) -> Literal["Running", "Succeeded", "Failed"]:
    """
    Query status of a calculation job
    Args:
        job_id (str): The ID of the calculation job
    Returns:
        status (str): One of "Running", "Succeeded" or "Failed"
    """
    trace_id, exec_id = job_id.split("/")
    with set_directory(trace_id):
        executor = load_job_info()["executor"] or executor
        _, executor = init_executor(executor)
        status = executor.query_status(exec_id)
        logger.info("Job %s status is %s" % (job_id, status))
    return status


def terminate_job(job_id: str, executor: Optional[dict] = None):
    """
    Terminate a calculation job
    Args:
        job_id (str): The ID of the calculation job
    """
    trace_id, exec_id = job_id.split("/")
    with set_directory(trace_id):
        executor = load_job_info()["executor"] or executor
        _, executor = init_executor(executor)
        executor.terminate(exec_id)
        logger.info("Job %s is terminated" % job_id)


def handle_input_artifacts(fn, kwargs, storage):
    storage_type, storage = init_storage(storage)
    sig = inspect.signature(fn)
    input_artifacts = {}
    for name, param in sig.parameters.items():
        if param.annotation is Path or (
            param.annotation is Optional[Path] and
                kwargs.get(name) is not None):
            uri = kwargs[name]
            scheme, key = parse_uri(uri)
            if scheme == storage_type:
                s = storage
            else:
                s = storage_dict[scheme]()
            path = s.download(key, "inputs/%s" % name)
            logger.info("Artifact %s downloaded to %s" % (
                uri, path))
            kwargs[name] = Path(path)
            input_artifacts[name] = {
                "storage_type": scheme,
                "uri": uri,
            }
        elif param.annotation is List[Path] or (
            param.annotation is Optional[List[Path]] and
                kwargs.get(name) is not None):
            uris = kwargs[name]
            new_paths = []
            for i, uri in enumerate(uris):
                scheme, key = parse_uri(uri)
                if scheme == storage_type:
                    s = storage
                else:
                    s = storage_dict[scheme]()
                dest_dir = Path("inputs") / name / f"item_{i:03d}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                path = s.download(key, str(dest_dir))
                new_paths.append(Path(path))
                logger.info("Artifact %s downloaded to %s" % (
                    uri, path))
            kwargs[name] = new_paths
            input_artifacts[name] = {
                "storage_type": storage_type,
                "uri": uris,
            }
        elif param.annotation is Dict[str, Path] or (
            param.annotation is Optional[Dict[str, Path]] and
                kwargs.get(name) is not None):
            uris_dict = kwargs[name]
            new_paths_dict = {}
            for key_name, uri in uris_dict.items():
                scheme, key = parse_uri(uri)
                if scheme == storage_type:
                    s = storage
                else:
                    s = storage_dict[scheme]()
                path = s.download(key, f"inputs/{name}/{key_name}")
                new_paths_dict[key_name] = Path(path)
                logger.info("Artifact %s (key=%s) downloaded to %s" % (
                    uri, key_name, path))
            kwargs[name] = new_paths_dict
            input_artifacts[name] = {
                "storage_type": storage_type,
                "uri": uris_dict,
            }
        elif param.annotation is Dict[str, List[Path]] or (
            param.annotation is Optional[Dict[str, List[Path]]] and
                kwargs.get(name) is not None):
            uris_dict = kwargs[name]
            new_paths_dict = {}
            for key_name, uris in uris_dict.items():
                new_paths = []
                for i, uri in enumerate(uris):
                    scheme, key = parse_uri(uri)
                    if scheme == storage_type:
                        s = storage
                    else:
                        s = storage_dict[scheme]()
                    dest_dir = Path("inputs") / name / key_name / f"item_{i:03d}"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    path = s.download(key, str(dest_dir))
                    new_paths.append(Path(path))
                    logger.info("Artifact %s (key=%s) downloaded to %s" % (
                        uri, key_name, path))
                new_paths_dict[key_name] = new_paths
            kwargs[name] = new_paths_dict
            input_artifacts[name] = {
                "storage_type": storage_type,
                "uri": uris_dict,
            }
    return kwargs, input_artifacts


def handle_output_artifacts(results, exec_id, storage):
    storage_type, storage = init_storage(storage)
    output_artifacts = {}
    if isinstance(results, dict):
        for name in results:
            if isinstance(results[name], Path):
                key = storage.upload("%s/outputs/%s" % (exec_id, name),
                                     results[name])
                uri = storage_type + "://" + key
                logger.info("Artifact %s uploaded to %s" % (
                    results[name], uri))
                results[name] = uri
                output_artifacts[name] = {
                    "storage_type": storage_type,
                    "uri": uri,
                }
            elif isinstance(results[name], list) and all(
                    isinstance(item, Path) for item in results[name]):
                new_uris = []
                for item in results[name]:
                    key = storage.upload("%s/outputs/%s" % (exec_id, name),
                                         item)
                    uri = storage_type + "://" + key
                    logger.info("Artifact %s uploaded to %s" % (
                        item, uri))
                    new_uris.append(uri)
                results[name] = new_uris
                output_artifacts[name] = {
                    "storage_type": storage_type,
                    "uri": new_uris,
                }
    return results, output_artifacts


# MCP does not regard Any as serializable in Python 3.12
# use Optional[Any] to work around
def get_job_results(job_id: str, executor: Optional[dict] = None,
                    storage: Optional[dict] = None):
    """
    Get results of a calculation job
    Args:
        job_id (str): The ID of the calculation job
    Returns:
        results (Any): results of the calculation job
    """
    trace_id, exec_id = job_id.split("/")
    with set_directory(trace_id):
        job_info = load_job_info()
        executor = job_info["executor"] or executor
        storage = job_info["storage"] or storage
        _, executor = init_executor(executor)
        results = executor.get_results(exec_id)
        results, output_artifacts = handle_output_artifacts(
            results, exec_id, storage)
        logger.info("Job %s result is %s" % (job_id, results))
    return JobResult(result=results, job_info={
        "output_artifacts": output_artifacts,
    }, tool_name=job_info["tool_name"])


annotation_map = {
    Path: str,
    Optional[Path]: Optional[str],
    List[Path]: List[str],
    Optional[List[Path]]: Optional[List[str]],
    Dict[str, Path]: Dict[str, str],
    Optional[Dict[str, Path]]: Optional[Dict[str, str]],
    Dict[str, List[Path]]: Dict[str, List[str]],
    Optional[Dict[str, List[Path]]]: Optional[Dict[str, List[str]]],
}


class SubmitResult(BaseModel):
    job_id: str
    extra_info: dict | None = None


def patch_mcp_close_connection():
    _mock_orig_handle_post_message = SseServerTransport.handle_post_message

    async def _mock_handle_post_message_with_close(self, scope, receive, send):
        async def _send(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers = [
                    (name, value)
                    for name, value in headers
                    if name.lower() != b"connection"
                ]
                headers.append((b"connection", b"close"))
                message["headers"] = headers
            elif message.get("type") == "http.response.body":
                message["more_body"] = False
            await send(message)
        await _mock_orig_handle_post_message(self, scope, receive, _send)

    if not getattr(SseServerTransport.handle_post_message,
                   "__patched_close__", False):
        SseServerTransport.handle_post_message = \
            _mock_handle_post_message_with_close
        SseServerTransport.handle_post_message.__patched_close__ = True


class CalculationMCPServer:
    def __init__(self, *args, preprocess_func=None, fastmcp_mode=False,
                 patch_close_connection=False, **kwargs):
        """
        Args:
            preprocess_func: The preprocess function for all tools
            fastmcp_mode: compatible for fastmcp.FastMCP
        """
        self.preprocess_func = preprocess_func
        self.fastmcp_mode = fastmcp_mode
        if patch_close_connection:
            patch_mcp_close_connection()
        self.mcp = FastMCP(*args, **kwargs)
        self.fn_metadata_map = {}

    def add_patched_tool(self, fn, new_fn, name, is_async=False, doc=None,
                         override_return_annotation=False):
        """patch the metadata of the tool"""
        context_kwarg = find_context_parameter(fn)
        func_arg_metadata = func_metadata(
            fn,
            skip_names=[context_kwarg] if context_kwarg is not None else [],
        )
        self.fn_metadata_map[name] = func_arg_metadata
        model_params = {}
        params = inspect.signature(fn, eval_str=True).parameters
        for n, annotation in \
                func_arg_metadata.arg_model.__annotations__.items():
            param = params[n]
            if param.annotation in annotation_map:
                model_params[n] = Annotated[
                    (annotation_map[param.annotation], Field())]
            else:
                model_params[n] = annotation
            if param.default is not inspect.Parameter.empty:
                model_params[n] = (model_params[n], param.default)
        for n, param in inspect.signature(new_fn).parameters.items():
            if n == "kwargs":
                continue
            model_params[n] = Annotated[(param.annotation, Field())]
            if param.default is not inspect.Parameter.empty:
                model_params[n] = (model_params[n], param.default)

        func_arg_metadata.arg_model = create_model(
            f"{fn.__name__}Arguments",
            __base__=ArgModelBase,
            **model_params,
        )
        if override_return_annotation:
            new_func_arg_metadata = func_metadata(new_fn)
            func_arg_metadata.output_model = new_func_arg_metadata.output_model
            func_arg_metadata.output_schema = \
                new_func_arg_metadata.output_schema
            func_arg_metadata.wrap_output = new_func_arg_metadata.wrap_output
        if self.fastmcp_mode and func_arg_metadata.wrap_output:
            # Only simulate behavior of fastmcp for output_schema
            func_arg_metadata.output_schema["x-fastmcp-wrap-result"] = True
        parameters = func_arg_metadata.arg_model.model_json_schema(
            by_alias=True)
        tool = Tool(
            fn=new_fn,
            name=name,
            description=doc or fn.__doc__ or "",
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            context_kwarg=context_kwarg,
            fn_metadata_map=self.fn_metadata_map,
        )
        self.mcp._tool_manager._tools[name] = tool

    def add_tool(self, fn, *args, **kwargs):
        tool = Tool.from_function(
            fn, *args, fn_metadata_map=self.fn_metadata_map, **kwargs)
        self.mcp._tool_manager._tools[tool.name] = tool
        return tool

    def tool(self, preprocess_func=None, create_workdir=None):
        # When create_workdir is None, do not create workdir when fn is async
        # and running locally to avoid chdir conflicts, create otherwise
        if preprocess_func is None:
            preprocess_func = self.preprocess_func

        def decorator(fn: Callable) -> Callable:
            def submit_job(executor: Optional[dict] = None,
                           storage: Optional[dict] = None,
                           **kwargs) -> SubmitResult:
                trace_id = datetime.today().strftime('%Y-%m-%d-%H:%M:%S.%f')
                logger.info("Job processing (Trace ID: %s)" % trace_id)
                if create_workdir is False:
                    workdir = "."
                else:
                    workdir = trace_id
                with set_directory(workdir):
                    if preprocess_func is not None:
                        executor, storage, kwargs = preprocess_func(
                            executor, storage, kwargs)
                    job = {
                        "tool_name": fn.__name__,
                        "executor": executor,
                        "storage": storage,
                    }
                    with open("job.json", "w") as f:
                        json.dump(job, f, indent=4)
                    kwargs, input_artifacts = handle_input_artifacts(
                        fn, kwargs, storage)
                    executor_type, executor = init_executor(executor)
                    res = executor.submit(fn, kwargs)
                    exec_id = res["job_id"]
                    job_id = "%s/%s" % (workdir, exec_id)
                    logger.info("Job submitted (ID: %s)" % job_id)
                result = SubmitResult(
                    job_id=job_id,
                    extra_info=res.get("extra_info"),
                )
                return JobResult(result=result, job_info={
                    "trace_id": trace_id,
                    "executor_type": executor_type,
                    "job_id": job_id,
                    "extra_info": res.get("extra_info"),
                    "input_artifacts": input_artifacts,
                })

            async def run_job(executor: Optional[dict] = None,
                              storage: Optional[dict] = None, **kwargs):
                context = self.mcp.get_context()
                trace_id = datetime.today().strftime('%Y-%m-%d-%H:%M:%S.%f')
                logger.info("Job processing (Trace ID: %s)" % trace_id)
                if preprocess_func is not None:
                    executor, storage, kwargs = preprocess_func(
                        executor, storage, kwargs)
                executor_type, executor = init_executor(executor)
                if create_workdir is False or (
                    create_workdir is None and inspect.iscoroutinefunction(fn)
                        and executor_type == "local"):
                    workdir = "."
                else:
                    workdir = trace_id
                with set_directory(workdir):
                    kwargs, input_artifacts = handle_input_artifacts(
                        fn, kwargs, storage)
                    res = await executor.async_run(
                        fn, kwargs, context, workdir)
                    exec_id = res["job_id"]
                    job_id = "%s/%s" % (workdir, exec_id)
                    results = res["result"]
                    results, output_artifacts = handle_output_artifacts(
                        results, exec_id, storage)
                    logger.info("Job %s result is %s" % (job_id, results))
                    await context.log(level="info", message="Job %s result is"
                                      " %s" % (job_id, results))
                return JobResult(result=results, job_info={
                    "trace_id": trace_id,
                    "executor_type": executor_type,
                    "job_id": job_id,
                    "extra_info": res.get("extra_info"),
                    "input_artifacts": input_artifacts,
                    "output_artifacts": output_artifacts,
                })

            self.add_patched_tool(fn, run_job, fn.__name__, is_async=True)
            self.add_patched_tool(
                fn, submit_job, "submit_" + fn.__name__, doc="Submit a job",
                override_return_annotation=True)
            self.add_tool(query_job_status)
            self.add_tool(terminate_job)
            self.add_tool(get_job_results)
            return fn
        return decorator

    @property
    def app(self):
        self.mcp.settings.stateless_http = True
        return self.mcp.streamable_http_app()

    def run(self, host=None, port=None, **kwargs):
        if os.environ.get("DP_AGENT_RUNNING_MODE") in ["1", "true"]:
            return

        async def health_check(request):
            return JSONResponse({"status": "ok"})

        self.mcp._custom_starlette_routes.append(
            Route(
                "/health",
                endpoint=health_check,
                methods=["GET"],
                name="health_check",
                include_in_schema=True,
            )
        )
        if host is not None:
            self.mcp.settings.host = host
        if port is not None:
            self.mcp.settings.port = port
        self.mcp.run(**kwargs)
