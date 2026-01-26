import asyncio
import importlib
import inspect
import io
import jsonpickle
import os
import psutil
import re
import sys
import time
import uuid
from multiprocessing import Process
from typing import Dict, Optional

from .base_executor import BaseExecutor
from ..utils import get_logger

DFLOW_ID_PATTERN = r"Workflow has been submitted \(ID: ([^,]*), UID: ([^)]*)\)"
DFLOW_LINK_PATTERN = r"Workflow link: (.*)"
logger = get_logger(__name__)


class Tee(io.TextIOBase):
    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)


def wrapped_fn(fn, kwargs, redirect_log=False):
    pid = os.getpid()
    # explicitly reload dflow config
    reload_dflow_config()
    if redirect_log:
        stdout = sys.stdout
        flog = open("%s.log" % pid, "w")
        sys.stdout = Tee(flog, stdout)
    try:
        if inspect.iscoroutinefunction(fn):
            result = asyncio.run(fn(**kwargs))
        else:
            result = fn(**kwargs)
    except Exception as e:
        with open("%s.err" % pid, "w") as f:
            f.write(str(e))
        raise e
    finally:
        if redirect_log:
            sys.stdout = stdout
            flog.close()
    with open("%s.txt" % pid, "w") as f:
        f.write(jsonpickle.dumps(result))


def reload_dflow_config():
    if "dflow.config" in sys.modules:
        config = sys.modules["dflow"].config
        s3_config = sys.modules["dflow"].s3_config
        s3_config["storage_client"] = None
        importlib.reload(sys.modules["dflow.config"])
        reload_bohrium = "dflow.plugins.bohrium" in sys.modules
        if reload_bohrium:
            bohrium_config = sys.modules["dflow.plugins.bohrium"].config
            importlib.reload(sys.modules["dflow.plugins.bohrium"])
        importlib.reload(sys.modules["dflow"])
        config.update(sys.modules["dflow"].config)
        sys.modules["dflow"].config = config
        s3_config.update(sys.modules["dflow"].s3_config)
        sys.modules["dflow"].s3_config = s3_config
        if reload_bohrium:
            bohrium_config.update(sys.modules["dflow.plugins.bohrium"].config)
            sys.modules["dflow.plugins.bohrium"].config = bohrium_config


class LocalExecutor(BaseExecutor):
    def __init__(self, env: Optional[Dict[str, str]] = None,
                 dflow: bool = False):
        """
        Execute the tool locally
        Args:
            env: The environmental variables at run time
            dflow: Wait until workflow submitted in submit method
        """
        self.env = env or {}
        self.dflow = dflow
        self.workflow_id = None

    def set_env(self):
        old_env = {}
        for k, v in self.env.items():
            if k in os.environ:
                old_env[k] = os.environ[k]
            os.environ[k] = v
        return old_env

    def recover_env(self, old_env):
        for k, v in self.env.items():
            if k in old_env:
                os.environ[k] = old_env[k]
            else:
                del os.environ[k]

    def submit(self, fn, kwargs):
        kwargs = self.prune_context(kwargs)
        os.environ["DP_AGENT_RUNNING_MODE"] = "1"
        old_env = self.set_env()
        params = {"fn": fn, "kwargs": kwargs}
        if self.dflow:
            params["redirect_log"] = True
        p = Process(target=wrapped_fn, kwargs=params)
        p.start()
        extra_info = {}
        if self.dflow:
            while True:
                alive = p.is_alive()
                if os.path.isfile("%s.log" % p.pid):
                    with open("%s.log" % p.pid, "r") as f:
                        log = f.read()
                    match_id = re.search(DFLOW_ID_PATTERN, log)
                    match_link = re.search(DFLOW_LINK_PATTERN, log)
                    if match_id and match_link:
                        wf_id = match_id.group(1)
                        self.workflow_id = wf_id
                        wf_uid = match_id.group(2)
                        wf_link = match_link.group(1)
                        extra_info["workflow_id"] = wf_id
                        extra_info["workflow_uid"] = wf_uid
                        extra_info["workflow_link"] = wf_link
                        break
                if not alive:
                    if os.path.isfile("%s.err" % p.pid):
                        with open("%s.err" % p.pid, "r") as f:
                            err_msg = f.read()
                    else:
                        err_msg = "No workflow submitted"
                    raise RuntimeError(err_msg)
                logger.info("Waiting workflow to be submitted")
                time.sleep(1)
        self.recover_env(old_env)
        return {"job_id": str(p.pid), "extra_info": extra_info}

    def query_status(self, job_id):
        try:
            p = psutil.Process(int(job_id))
            if p.status() not in ["zombie", "dead"]:
                return "Running"
        except psutil.NoSuchProcess:
            pass

        if os.path.isfile("%s.txt" % job_id):
            return "Succeeded"
        else:
            return "Failed"

    def terminate(self, job_id):
        if self.workflow_id is not None:
            try:
                from dflow import Workflow
                wf = Workflow(id=self.workflow_id)
                wf.terminate()
            except Exception as e:
                logger.error(f"Failed to terminate workflow: {e}")
        try:
            p = psutil.Process(int(job_id))
            p.terminate()
        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")

    def get_results(self, job_id):
        if os.path.isfile("%s.txt" % job_id):
            with open("%s.txt" % job_id, "r") as f:
                return jsonpickle.loads(f.read())
        elif os.path.isfile("%s.err" % job_id):
            with open("%s.err" % job_id, "r") as f:
                err_msg = f.read()
            raise RuntimeError(err_msg)
        return {}

    async def async_run(self, fn, kwargs, context, trace_id):
        os.environ["DP_AGENT_RUNNING_MODE"] = "1"
        old_env = self.set_env()
        try:
            # explicitly reload dflow config
            reload_dflow_config()
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = fn(**kwargs)
        finally:
            self.recover_env(old_env)
        return {
            "job_id": str(uuid.uuid4()),
            "result": result,
        }
