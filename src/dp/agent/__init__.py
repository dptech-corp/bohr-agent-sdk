import os
RUNNING_MODE = os.environ.get("DP_AGENT_RUNNING_MODE") in ["1", "true"]
if not RUNNING_MODE:
    from .cli.cli import main

__all__ = ["main"]
