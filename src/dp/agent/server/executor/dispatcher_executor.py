import importlib
import inspect
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import jsonpickle
from dpdispatcher import Machine, Resources, Task, Submission
from dpdispatcher.utils.job_status import JobStatus

from .base_executor import BaseExecutor
from .... import __path__

config = {
    "username": os.environ.get("BOHRIUM_USERNAME", ""),
    "password": os.environ.get("BOHRIUM_PASSWORD", ""),
    "project_id": os.environ.get("BOHRIUM_PROJECT_ID", ""),
    "access_key": os.environ.get("BOHRIUM_ACCESS_KEY", ""),
    "app_key": os.environ.get("BOHRIUM_APP_KEY", "agent"),
    "bohrium_url": os.environ.get("BOHRIUM_BOHRIUM_URL",
                                  "https://bohrium.dp.tech"),
}
logger = logging.getLogger(__name__)


def get_source_code(fn):
    source_lines, start_line = inspect.getsourcelines(fn)
    source_file = inspect.getsourcefile(fn)
    with open(source_file, "r", encoding="utf-8") as fd:
        pre_lines = fd.readlines()[:start_line - 1]
    return "".join(pre_lines + source_lines) + "\n"


def get_func_def_script(fn):
    script = ""
    packages = []
    fn_name = fn.__name__
    module_name = fn.__module__
    module = sys.modules[module_name]
    if getattr(module, fn_name, None) is not fn:
        # cannot import from module, maybe a local function
        import cloudpickle
        packages.extend(cloudpickle.__path__)
        script += "import cloudpickle\n"
        script += "%s = cloudpickle.loads(%s)\n" % \
            (fn_name, cloudpickle.dumps(fn))
    elif module_name in ["__main__", "__mp_main__"]:
        if hasattr(module, "__file__"):
            name = os.path.splitext(os.path.basename(module.__file__))[0]
            if getattr(module, "__package__", None):
                package = module.__package__
                package_name = package.split('.')[0]
                module = importlib.import_module(package_name)
                packages.extend(module.__path__)
                script += "from %s.%s import %s\n" % (
                    package, name, fn_name)
            else:
                packages.append(module.__file__)
                script += "from %s import %s\n" % (name, fn_name)
        else:
            # cannot get file of __main__, maybe in the interactive mode
            import cloudpickle
            packages.extend(cloudpickle.__path__)
            script += "import cloudpickle\n"
            script += "%s = cloudpickle.loads(%s)\n" % \
                (fn_name, cloudpickle.dumps(fn))
    else:
        package_name = module_name.split('.')[0]
        module = importlib.import_module(package_name)
        if hasattr(module, "__path__"):
            packages.extend(module.__path__)
        elif hasattr(module, "__file__"):
            packages.append(module.__file__)
        script += "from %s import %s\n" % (module_name, fn_name)
    return script, packages


class DispatcherExecutor(BaseExecutor):
    def __init__(
            self,
            machine=None,
            resources=None,
            python_packages=None,
            python_executable="python3",
    ):
        """Use DPDispatcher to execute the tool
        Refer to https://docs.deepmodeling.com/projects/dpdispatcher.

        Args:
            machine: The machine configuration of DPDispatcher
            resources: The resources configuration of DPDispatcher
            python_packages: Additional python packages uploaded to runtime
                environment
            python_executable: Python executable path for running the tool
        """
        self.machine = machine or {}
        self.resources = resources or {}
        self.python_packages = python_packages or []
        self.python_packages.extend(__path__)
        self.python_packages.extend(jsonpickle.__path__)
        self.python_executable = python_executable
        self.set_defaults()

    def set_defaults(self):
        self.machine["local_root"] = "."
        if self.machine.get("context_type") == "Bohrium":
            remote_profile = self.machine["remote_profile"]
            if "email" not in remote_profile:
                remote_profile["email"] = config["username"]
            if "password" not in remote_profile:
                remote_profile["password"] = config["password"]
            if "program_id" not in remote_profile:
                remote_profile["program_id"] = int(config["project_id"])
        elif self.machine.get("context_type") == "OpenAPI":
            remote_profile = self.machine["remote_profile"]
            if "access_key" not in remote_profile:
                remote_profile["access_key"] = config["access_key"]
            if "project_id" not in remote_profile:
                remote_profile["project_id"] = int(config["project_id"])
            if "app_key" not in remote_profile:
                remote_profile["app_key"] = config["app_key"]
        if "group_size" not in self.resources:
            self.resources["group_size"] = 1
        if "envs" not in self.resources:
            self.resources["envs"] = {}
        self.resources["envs"]["DP_AGENT_RUNNING_MODE"] = "1"

    def submit(self, fn, kwargs):
        kwargs = self.prune_context(kwargs)
        script = ""
        fn_name = fn.__name__
        func_def_script, packages = get_func_def_script(fn)
        self.python_packages.extend(packages)

        script += "import asyncio, jsonpickle, os, shutil\n"
        script += "from pathlib import Path\n\n"
        script += "if __name__ == \"__main__\":\n"
        script += "    cwd = os.getcwd()\n"
        script += "    kwargs = jsonpickle.loads(%s)\n" % repr(
            jsonpickle.dumps(kwargs))
        script += "    try:\n"
        for line in func_def_script.splitlines():
            if line:
                script += "        " + line + "\n"
        if inspect.iscoroutinefunction(fn):
            script += "        results = asyncio.run(%s(**kwargs))\n" % fn_name
        else:
            script += "        results = %s(**kwargs)\n" % fn_name
            script += "        result_dir = None\n"
            script += "        import uuid\n"
            script += "        if isinstance(results, dict):\n"
            script += "            for name in results:\n"
            script += "                if isinstance(results[name], Path):\n"
            script += "                    if not results[name].absolute().is_relative_to(cwd):\n"
            script += "                        if result_dir is None:\n"
            script += "                            result_dir = Path('result_files_dir_' + str(uuid.uuid4()))\n"
            script += "                            result_dir.mkdir(parents=True, exist_ok=True)\n"
            script += "                        dest_path = result_dir / results[name].absolute().relative_to('/')\n"
            script += "                        dest_path.parent.mkdir(parents=True, exist_ok=True)\n"
            script += "                        if results[name].is_file():\n"
            script += "                            shutil.copy2(results[name], dest_path)\n"
            script += "                        elif results[name].is_dir():\n"
            script += "                            shutil.copytree(results[name], dest_path, dirs_exist_ok=True)\n"
            script += "                        results[name] = dest_path.absolute().relative_to(cwd)\n"
            script += "                    else:\n"
            script += "                        results[name] = results[name].absolute().relative_to(cwd)\n"
        script += "    except Exception as e:\n"
        script += "        os.chdir(cwd)\n"
        script += "        with open('err', 'w') as f:\n"
        script += "            f.write(str(e))\n"
        script += "        raise e\n"
        script += "    os.chdir(cwd)\n"
        script += "    with open('results.txt', 'w') as f:\n"
        script += "        f.write(jsonpickle.dumps(results))\n"
        with open("script.py", "w") as f:
            f.write(script)

        forward_files = ["script.py"]
        for package in self.python_packages:
            target = os.path.basename(package)
            if os.path.abspath(package) != os.path.abspath(target):
                copy_method = os.getenv("DISPATCHER_EXECUTOR_COPY_METHOD",
                                        "symlink")
                if copy_method == "symlink":
                    if os.path.islink(target):
                        os.remove(target)
                    os.symlink(package, target)
                elif copy_method == "copy":
                    if os.path.isfile(package):
                        shutil.copy(package, target)
                    elif os.path.isdir(package):
                        shutil.copytree(package, target)
            if target not in forward_files:
                forward_files.append(target)
        for value in kwargs.values():
            if isinstance(value, Path):
                forward_files.append(str(value))

        task = {
            "task_work_path": "./",
            "outlog": "log",
            "errlog": "log",
            "command": self.python_executable + " script.py",
            "forward_files": forward_files,
        }

        if self.machine.get("context_type") == "Bohrium":
            self.machine["remote_profile"]["input_data"]["job_name"] = fn_name
        elif self.machine.get("context_type") == "OpenAPI":
            self.machine["remote_profile"]["job_name"] = fn_name
        # ensure submitting a new job
        self.resources["envs"]["SUBMISSION_TIMESTAMP"] = str(time.time())

        machine = Machine.load_from_dict(self.machine)
        resources = Resources.load_from_dict(self.resources)
        task = Task.load_from_dict(task)
        submission = Submission(
            work_base='.', machine=machine, resources=resources,
            task_list=[task])
        submission.run_submission(exit_on_submit=True)
        res = {"job_id": submission.submission_hash}
        if self.machine.get("context_type") == "Bohrium":
            job_id = submission.belonging_jobs[0].job_id
            bohr_job_id, bohr_group_id = job_id.split(":job_group_id:")
            extra_info = {
                "bohr_job_id": bohr_job_id,
                "bohr_group_id": bohr_group_id,
                "job_link": f"{config['bohrium_url']}/jobs/detail/%s" %
                            bohr_job_id,
            }
            logger.info(extra_info)
            res["extra_info"] = extra_info
        elif self.machine.get("context_type") == "OpenAPI":
            bohr_job_id = submission.belonging_jobs[0].job_id
            extra_info = {
                "bohr_job_id": bohr_job_id,
                "job_link": f"{config['bohrium_url']}/jobs/detail/%s" %
                            bohr_job_id,
            }
            logger.info(extra_info)
            res["extra_info"] = extra_info
        return res

    def query_status(self, job_id):
        machine = Machine.load_from_dict(self.machine)
        content = machine.context.read_file(job_id + ".json")
        submission = Submission.deserialize(
            submission_dict=json.loads(content))
        submission.update_submission_state()
        if not submission.check_all_finished() and not any(
            job.job_state in [JobStatus.terminated, JobStatus.unknown,
                              JobStatus.unsubmitted]
                for job in submission.belonging_jobs):
            return "Running"
        try:
            submission.run_submission(exit_on_submit=True)
        except Exception as e:
            logger.error(e)
            return "Failed"
        if submission.check_all_finished():
            if os.path.isfile("results.txt"):
                return "Succeeded"
            else:
                return "Failed"
        else:
            return "Running"

    def terminate(self, job_id):
        machine = Machine.load_from_dict(self.machine)
        content = machine.context.read_file(job_id + ".json")
        submission = Submission.deserialize(
            submission_dict=json.loads(content))
        submission.remove_unfinished_tasks()

    def get_results(self, job_id):
        if os.path.isfile("results.txt"):
            with open("results.txt", "r") as f:
                return jsonpickle.loads(f.read())
        elif os.path.isfile("err"):
            with open("err", "r") as f:
                err_msg = f.read()
            raise RuntimeError(err_msg)
        return {}
