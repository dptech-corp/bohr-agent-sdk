from .base_executor import BaseExecutor
from .dispatcher_executor import DispatcherExecutor
from .local_executor import LocalExecutor

__all__ = ["BaseExecutor"]
executor_dict = {
    "dispatcher": DispatcherExecutor,
    "local": LocalExecutor,
}
