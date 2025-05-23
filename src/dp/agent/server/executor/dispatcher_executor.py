import importlib
import inspect
import json
import logging
import os
import shutil
import time
from pathlib import Path

import jsonpickle
from dpdispatcher import Machine, Resources, Task, Submission

from .base_executor import BaseExecutor
from .... import __path__
config = {
    "username": os.environ.get("BOHRIUM_USERNAME", ""),
    "password": os.environ.get("BOHRIUM_PASSWORD", ""),
    "project_id": os.environ.get("BOHRIUM_PROJECT_ID", ""),
}
logger = logging.getLogger(__name__)


def get_source_code(fn):
    source_lines, start_line = inspect.getsourcelines(fn)
    source_file = inspect.getsourcefile(fn)
    with open(source_file, "r", encoding="utf-8") as fd:
        pre_lines = fd.readlines()[:start_line-1]
    return "".join(pre_lines + source_lines) + "\n"


class DispatcherExecutor(BaseExecutor):
    def __init__(
        self,
        machine=None,
        resources=None,
        python_packages=None,
        pip_packages=None,
        python_executable="python3",
        DEFAULT_FORWARD_DIR = "dispatcher_assets",
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
        self.pip_packages = pip_packages or []
        self.DEFAULT_FORWARD_DIR = DEFAULT_FORWARD_DIR
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
        if "group_size" not in self.resources:
            self.resources["group_size"] = 1
        if "envs" not in self.resources:
            self.resources["envs"] = {}
        self.resources["envs"]["DP_AGENT_RUNNING_MODE"] = "1"

    def submit(self, fn, kwargs):
        script = ""
        script += "import os, sys, subprocess\n"
        if self.pip_packages:
            script += "subprocess.check_call([sys.executable, '-m', 'pip', 'install', "
            for pip_package in self.pip_packages:
                script += "'%s', " % pip_package
            script += "])\n"


        fn_name = fn.__name__
        module_name = fn.__module__
        if module_name in ["__main__", "__mp_main__"]:
            script += get_source_code(fn)
        else:
            package_name = module_name.split('.')[0]
            module = importlib.import_module(package_name)
            if hasattr(module, "__path__"):
                self.python_packages.extend(module.__path__)
            elif hasattr(module, "__file__"):
                self.python_packages.append(module.__file__)
            script += "from %s import %s\n" % (module_name, fn_name)

        script += "import jsonpickle\n\n"
        script += "if __name__ == \"__main__\":\n"
        script += "    results = %s(**jsonpickle.loads(r'''%s'''))\n" % (
            fn_name, jsonpickle.dumps(kwargs))
        script += "    with open('results.txt', 'w') as f:\n"
        script += "        f.write(jsonpickle.dumps(results))\n"
        with open("script.py", "w") as f:
            f.write(script)

        forward_files = ["script.py"]
        forward_files.extend(self.DEFAULT_FORWARD_DIR)
        for package in self.python_packages:
            target = os.path.basename(package)
            if os.path.abspath(package) != os.path.abspath(target):
                if os.path.islink(target):
                    os.remove(target)
                os.symlink(package, target)
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
        # ensure submitting a new job
        self.resources["envs"]["SUBMISSION_TIMESTAMP"] = str(time.time())

        machine = Machine.load_from_dict(self.machine)
        resources = Resources.load_from_dict(self.resources)
        task = Task.load_from_dict(task)
        submission = Submission(
            work_base='.', machine=machine, resources=resources,
            task_list=[task])
        submission.run_submission(exit_on_submit=True)
        return submission.submission_hash

    def query_status(self, job_id):
        machine = Machine.load_from_dict(self.machine)
        content = machine.context.read_file(job_id + ".json")
        submission = Submission.deserialize(
            submission_dict=json.loads(content))
        submission.update_submission_state()
        if not submission.check_all_finished():
            return "Running"
        try:
            submission.run_submission(exit_on_submit=True)
        except Exception as e:
            logger.error(e)
            return "Failed"
        if os.path.isfile("results.txt"):
            shutil.move("results.txt", "%s.txt" % job_id)
            return "Succeeded"
        else:
            return "Failed"

    def terminate(self, job_id):
        machine = Machine.load_from_dict(self.machine)
        content = machine.context.read_file(job_id + ".json")
        submission = Submission.deserialize(
            submission_dict=json.loads(content))
        submission.remove_unfinished_tasks()

    def get_results(self, job_id):
        if not os.path.isfile("%s.txt" % job_id):
            machine = Machine.load_from_dict(self.machine)
            content = machine.context.read_file(job_id + ".json")
            submission = Submission.deserialize(
                submission_dict=json.loads(content))
            submission.run_submission(exit_on_submit=True)
            if os.path.isfile("results.txt"):
                shutil.move("results.txt", "%s.txt" % job_id)
        if os.path.isfile("%s.txt" % job_id):
            with open("%s.txt" % job_id, "r") as f:
                return jsonpickle.loads(f.read())
        return None
