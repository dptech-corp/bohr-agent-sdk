import os
RUNNING_MODE = os.environ.get("DP_AGENT_RUNNING_MODE") in ["1", "true"]


class FakeCalculationMCPServer:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def run(self, *args, **kwargs):
        return


if RUNNING_MODE:
    CalculationMCPServer = FakeCalculationMCPServer
else:
    from .calculation_mcp_server import CalculationMCPServer

__all__ = ["CalculationMCPServer"]
