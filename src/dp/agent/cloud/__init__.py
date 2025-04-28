"""
Cloud service module for science-agent-sdk.
This module provides cloud service functionality for device control and monitoring.
"""

from .mqtt import MQTTCloud, get_mqtt_cloud_instance
from .mcp import mcp, get_mcp_instance

__all__ = [
    'mcp',  # 全局MCP实例
    'get_mcp_instance',  # 获取全局MCP实例的函数
    'MQTTCloud',  # MQTT客户端类
    'get_mqtt_cloud_instance',  # 获取全局MQTT客户端实例的函数
] 