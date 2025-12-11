#!/usr/bin/env python3
"""
Agent WebSocket 服务器 - 主入口文件
使用 Session 运行 rootagent，并通过 WebSocket 与前端通信
"""

import os
import sys
import warnings

# 忽略 paramiko 的加密算法弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")

# Add user working directory to Python path first
user_working_dir = os.environ.get('USER_WORKING_DIR')
if user_working_dir and user_working_dir not in sys.path:
    sys.path.insert(0, user_working_dir)

# Add UI template directory to Python path for config imports
ui_template_dir = os.environ.get('UI_TEMPLATE_DIR')
if ui_template_dir and ui_template_dir not in sys.path:
    sys.path.insert(0, ui_template_dir)

import uvicorn
from server.app import create_app
from server.utils import check_port_available
from config.agent_config import agentconfig


if __name__ == "__main__":
    print("🚀 启动 Agent WebSocket 服务器...")
    
    # 统一使用 server 配置
    server_config = agentconfig.config.get('server', {})
    port = server_config.get('port', 8000)
    # host 数组中的第一个作为显示用
    hosts = server_config.get('host', ['localhost'])
    display_host = hosts[0] if isinstance(hosts, list) else hosts
    
    
    # 创建应用
    app = create_app()
    
    print("📡 使用 Session 模式运行 rootagent")
    print(f"🌐 服务器地址: http://{display_host}:{port}")
    print(f"🔌 WebSocket 端点: ws://{display_host}:{port}/ws")
    print("🛑 使用 Ctrl+C 优雅关闭服务器")
    
    # uvicorn 始终监听 0.0.0.0 以支持所有配置的主机
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",  # 使用 info 级别，过滤掉 warning
        access_log=False,  # 禁用访问日志，减少噪音
        ws_ping_interval=20,  # Send WebSocket ping every 20s to keep connection alive
        ws_ping_timeout=30,   # Disconnect if no pong received within 30s
        # 添加自定义的日志配置
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"]
            },
            "loggers": {
                "uvicorn.error": {
                    "level": "ERROR"
                },
                "uvicorn.access": {
                    "handlers": [],
                    "propagate": False
                }
            }
        }
    )