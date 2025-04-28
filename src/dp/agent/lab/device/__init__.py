"""
Device module for science-agent-sdk.
This module provides base device functionality.
"""

from .device import Device, action
from .types import BaseParams, ActionResult, SuccessResult, ErrorResult

__all__ = [
    'Device',
    'action',
    'BaseParams',
    'ActionResult',
    'SuccessResult',
    'ErrorResult',
] 