import inspect
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.func_metadata import _get_typed_signature

from .executor import executor_dict
from .storage import storage_dict
from .utils import get_metadata

logger = logging.getLogger(__name__)


def parse_uri(uri):
    scheme = urlparse(uri).scheme
    if scheme == "":
        key = uri
        scheme = "local"
    else:
        key = uri[len(scheme)+3:]
    return scheme, key


def init_storage(storage_config: Optional[dict] = None):
    if storage_config is None:
        storage_config = {"type": "local"}
    storage_config = storage_config.copy()
    storage_type = storage_config.pop("type")
    storage = storage_dict[storage_type](**storage_config)
    return storage_type, storage


def init_executor(executor_config: Optional[dict] = None):
    if executor_config is None:
        executor_config = {"type": "local"}
    executor_config = executor_config.copy()
    executor_type = executor_config.pop("type")
    return executor_dict[executor_type](**executor_config)


def query_job_status(job_id: str, executor: Optional[dict] = None
                     ) -> Literal["Running", "Succeeded", "Failed"]:
    """
    Query status of a calculation job
    Args:
        job_id (str): The ID of the calculation job
    Returns:
        status (str): One of "Running", "Succeeded" or "Failed"
    """
    executor = init_executor(executor)
    status = executor.query_status(job_id)
    logger.info("Job %s status is %s" % (job_id, status))
    return status


def terminate_job(job_id: str, executor: Optional[dict] = None):
    """
    Terminate a calculation job
    Args:
        job_id (str): The ID of the calculation job
    """
    executor = init_executor(executor)
    executor.terminate(job_id)
    logger.info("Job %s is terminated" % job_id)


def get_job_results(job_id: str, executor: Optional[dict] = None,
                    storage: Optional[dict] = None) -> dict:
    """
    Get results of a calculation job
    Args:
        job_id (str): The ID of the calculation job
    Returns:
        results (dict): results of the calculation job
    """
    storage_type, storage = init_storage(storage)
    executor = init_executor(executor)
    results = executor.get_results(job_id)
    prefix = str(uuid.uuid4())
    for name in results:
        if isinstance(results[name], Path):
            key = storage.upload("%s/outputs/%s" % (prefix, name),
                                 results[name])
            uri = storage_type + "://" + key
            logger.info("Artifact %s uploaded to %s" % (
                results[name], uri))
            results[name] = uri
    logger.info("Job %s results is %s" % (job_id, results))
    return results

#lh add 
def safe_openai_schema(schema: dict) -> dict:
    """
    清理 JSON Schema，使其兼容 OpenAI Function Calling：
    - 删除 default（anyOf/oneOf 中不允许）
    - 添加 additionalProperties: false
    - 移除 executor 和 storage 字段（避免必须添加到 required 中）
    """
    REMOVE_KEYS = {"executor", "storage"}

    def is_optional(prop: dict) -> bool:
        return (
            isinstance(prop, dict)
            and "anyOf" in prop
            and any(p.get("type") == "null" for p in prop["anyOf"] if isinstance(p, dict))
        )

    def clean(s):
        if isinstance(s, dict):
            # 删除 default
            if ("anyOf" in s or "oneOf" in s) and "default" in s:
                del s["default"]

            # 给 object 添加 additionalProperties
            if s.get("type") == "object":
                s.setdefault("additionalProperties", False)

                # 移除指定字段
                if "properties" in s:
                    for key in list(s["properties"].keys()):
                        if key in REMOVE_KEYS:
                            del s["properties"][key]

                # 自动构建 required（排除已被移除的字段）
                if "properties" in s:
                    s["required"] = [
                        k for k, v in s["properties"].items()
                        if not is_optional(v)
                    ]
                    if not s["required"]:
                        s.pop("required", None)

            # 处理 anyOf/oneOf 中的对象类型
            for key in ["anyOf", "oneOf"]:
                if key in s:
                    for entry in s[key]:
                        if isinstance(entry, dict) and entry.get("type") == "object":
                            entry.setdefault("additionalProperties", False)

            # 递归
            for v in s.values():
                clean(v)

        elif isinstance(s, list):
            for item in s:
                clean(item)

        return s

    return clean(schema)

class CalculationMCPServer:
    def __init__(self, *args, **kwargs):
        self.mcp = FastMCP(*args, **kwargs)

    #lh add 
    def _add_and_patch_builtin(self, fn): 
        self.mcp.add_tool(fn)
        tool = self.mcp._tool_manager.get_tool(fn.__name__)
        tool.parameters = safe_openai_schema(tool.parameters)

    def add_patched_tool(self, fn, new_fn, name):
        self.mcp.add_tool(fn, name=name)
        # replace the function of the tool
        tool = self.mcp._tool_manager.get_tool(name)
        tool.fn = new_fn
        # patch the metadata of the tool
        # combine parameters
        parameters = []
        for param in _get_typed_signature(fn).parameters.values():
            if param.annotation is Path:
                parameters.append(inspect.Parameter(
                    name=param.name, default=param.default,
                    annotation=str, kind=param.kind))
            elif param.annotation is Optional[Path]:
                parameters.append(inspect.Parameter(
                    name=param.name, default=param.default,
                    annotation=Optional[str], kind=param.kind))
            else:
                parameters.append(param)
        for param in _get_typed_signature(new_fn).parameters.values():
            if param.name != "kwargs":
                parameters.append(param)
        func_arg_metadata = get_metadata(
            name,
            parameters=parameters,
            skip_names=[tool.context_kwarg]
            if tool.context_kwarg is not None else [],
            globalns=getattr(fn, "__globals__", {})
        )
        json_schema = func_arg_metadata.arg_model.model_json_schema()
        json_schema = safe_openai_schema(json_schema)
        tool.fn_metadata = func_arg_metadata
        tool.parameters = json_schema

    def tool(self):
        def decorator(fn: Callable) -> Callable:
            def submit_job(executor: Optional[dict] = None,
                           storage: Optional[dict] = None, **kwargs):
                storage_type, storage = init_storage(storage)
                sig = inspect.signature(fn)
                for name, param in sig.parameters.items():
                    if param.annotation is Path or (
                        param.annotation is Optional[Path] and
                            kwargs.get(name) is not None):
                        uri = kwargs[name]
                        scheme, key = parse_uri(uri)
                        assert scheme == storage_type
                        path = storage.download(key, "inputs/%s" % name)
                        logger.info("Artifact %s downloaded to %s" % (
                            uri, path))
                        kwargs[name] = Path(path)
                executor = init_executor(executor)
                job_id = executor.submit(fn, kwargs)
                logger.info("Job submitted (ID: %s)" % job_id)
                return job_id

            def run_job(executor: Optional[dict] = None,
                        storage: Optional[dict] = None, **kwargs):
                job_id = submit_job(
                    executor=executor, storage=storage, **kwargs)
                while True:
                    status = query_job_status(job_id, executor=executor)
                    if status != "Running":
                        break
                    time.sleep(4)
                if status == "Succeeded":
                    return get_job_results(
                        job_id, executor=executor, storage=storage)
                elif status == "Failed":
                    raise RuntimeError("Job failed")

            self.add_patched_tool(fn, run_job, fn.__name__)
            self.add_patched_tool(fn, submit_job, "submit_" + fn.__name__)
            #lh add 
            for builtin in (query_job_status, terminate_job, get_job_results):
                self._add_and_patch_builtin(builtin)
            return fn
        return decorator

    def run(self, **kwargs):
        self.mcp.run(**kwargs)
