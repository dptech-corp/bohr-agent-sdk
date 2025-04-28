from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal


class BaseExecutor(ABC):
    @abstractmethod
    def submit(self, fn: Callable, kwargs: dict) -> str:
        pass

    @abstractmethod
    def query_status(self, job_id: str) -> Literal[
            "Running", "Succeeded", "Failed"]:
        pass

    @abstractmethod
    def terminate(self, job_id: str) -> None:
        pass

    @abstractmethod
    def get_results(self, job_id: str) -> dict:
        pass
