import os
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict

import sys

from dp.agent.server.calculation_mcp_server import CalculationMCPServer

mcp = CalculationMCPServer("Demo")

@mcp.tool()
def add(a:int,b:int):
    """
    计算a+b
    返回计算结果
    """
    return {"result":a+b}

import jsonpickle

if __name__ == "__main__":
    results = add(**jsonpickle.loads(r'''{"a": 1, "b": 3}'''))
    with open('results.txt', 'w') as f:
        f.write(jsonpickle.dumps(results))
