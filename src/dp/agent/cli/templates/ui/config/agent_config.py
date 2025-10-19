"""
Agent Configuration Loader

This module provides a centralized configuration system for different agent implementations.
To switch between different agents, modify the agent-config.json file.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import importlib
import importlib.util

# 配置日志输出到文件
# 使用相对于项目根目录的路径或环境变量配置
log_file_path = os.environ.get('WEBSOCKET_LOG_PATH', './websocket.log')
# 只在没有配置过的情况下配置
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8', mode='a'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
logger = logging.getLogger(__name__)

class AgentConfig:
    def __init__(self, config_path: str = None):
        # 优先使用环境变量中的配置路径
        if config_path is None:
            env_path = os.environ.get('AGENT_CONFIG_PATH', 'config/agent-config.json')
            self.config_path = Path(env_path)
        else:
            self.config_path = Path(config_path)
        
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        logger.info(f"📦 加载配置文件: {self.config_path}")
        
        if not self.config_path.exists():
            logger.warning(f"⚠️ 配置文件不存在: {self.config_path}")
            logger.info("🔄 使用默认配置")
            # Fallback to default config if file doesn't exist
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info("✅ 配置文件加载成功")
                logger.debug(f"  Agent名称: {config.get('agent', {}).get('name')}")
                logger.debug(f"  Agent模块: {config.get('agent', {}).get('module')}")
                logger.debug(f"  Root Agent: {config.get('agent', {}).get('rootAgent')}")
                return config
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析错误: {e}")
            logger.info("🔄 使用默认配置")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ 读取配置文件失败: {e}")
            logger.info("🔄 使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default configuration for Agent"""
        # 使用环境变量提供的端口，如果没有则使用默认值
        default_port = int(os.environ.get('AGENT_SERVER_PORT', '50002'))
        
        return {
            "agent": {
                "name": "My Agent",
                "description": "Agent",
                "welcomeMessage": "welcome to chat with me",
                "module": "agent.subagent",
                "rootAgent": "rootagent"
            },
            "ui": {
                "title": "Agent",
                "features": {
                    "showFileExplorer": True,
                    "showSessionList": True
                }
            },
            "files": {
                "watchDirectories": ["./output"]
            },
            "server": {
                "host": ["localhost", "127.0.0.1"],
                "port": default_port
            }
        }
    
    def get_agent(self, ak: str = None, app_key: str = None, project_id: int = None):
        """Dynamically import and return the configured agent
        
        Args:
            ak: Optional access key to pass to the agent
            app_key: Optional app key to pass to the agent
            project_id: Optional project ID to pass to the agent
        """
        agentconfig = self.config.get("agent", {})
        module_path = agentconfig.get("module", "agent.subagent")
        agentname = agentconfig.get("rootAgent", "rootagent")
        
        logger.info(f"🤖 加载 Agent")
        logger.debug(f"  模块路径: {module_path}")
        logger.debug(f"  Agent名称: {agentname}")
        logger.debug(f"  AK: {'有' if ak else '无'}")
        logger.debug(f"  App Key: {'有' if app_key else '无'}")
        logger.debug(f"  Project ID: {project_id}")
        
        try:
            # 检查是否是文件路径（包含 / 或 \ 或以 .py 结尾）
            if '/' in module_path or '\\' in module_path or module_path.endswith('.py'):
                # 作为文件路径处理
                file_path = Path(module_path)
                
                # 如果是相对路径，基于用户工作目录解析
                if not file_path.is_absolute():
                    user_working_dir = os.environ.get('USER_WORKING_DIR', os.getcwd())
                    file_path = Path(user_working_dir) / file_path
                
                logger.debug(f"  解析文件路径: {file_path}")
                
                # 确保文件存在
                if not file_path.exists():
                    logger.error(f"❌ Agent模块文件不存在: {file_path}")
                    raise ImportError(f"Agent module file not found: {file_path}")
                
                logger.info(f"📄 从文件加载: {file_path}")
                
                # 从文件路径创建唯一的模块名，包含路径信息避免冲突
                # 例如: /path/to/agent.py -> path_to_agent
                module_name = str(file_path).replace('/', '_').replace('\\', '_').replace('.py', '').replace('.', '_')
                # 确保模块名是有效的 Python 标识符
                module_name = 'agent_' + module_name.strip('_')
                
                # 使用 importlib.util 从文件加载模块
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module from file: {file_path}")
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module  # 可选：将模块添加到 sys.modules
                spec.loader.exec_module(module)
                logger.info(f"✅ 模块加载成功: {module_name}")
            else:
                # 作为模块路径处理（原有逻辑）
                logger.debug(f"  作为 Python模块导入: {module_path}")
                module = importlib.import_module(module_path)
                logger.info(f"✅ 模块导入成功: {module_path}")
            
            # 检查是否有 create_agent 函数（推荐的方式）
            if hasattr(module, 'create_agent'):
                logger.info("🔧 使用 create_agent 函数创建 Agent")
                # 使用工厂函数创建新的 agent 实例
                # 检查函数接受哪些参数
                import inspect
                sig = inspect.signature(module.create_agent)
                params = {}
                
                # 只传递函数签名中存在的参数
                if 'ak' in sig.parameters:
                    params['ak'] = ak
                if 'app_key' in sig.parameters:
                    params['app_key'] = app_key
                if 'project_id' in sig.parameters:
                    params['project_id'] = project_id
                
                agent = module.create_agent(**params)
                logger.info(f"✅ Agent 创建成功: {type(agent).__name__}")
                return agent
            else:
                # 后向兼容：直接返回模块级别的 agent
                logger.info(f"🔍 查找模块属性: {agentname}")
                if hasattr(module, agentname):
                    agent = getattr(module, agentname)
                    logger.info(f"✅ 找到 Agent: {type(agent).__name__}")
                    return agent
                else:
                    logger.error(f"❌ 模块 {module_path} 中没有找到 {agentname}")
                    logger.debug(f"  可用属性: {dir(module)}")
                    raise AttributeError(f"模块 {module_path} 中没有 {agentname}")
        except ImportError as e:
            logger.error(f"❌ 导入错误: {e}")
            raise ImportError(f"Failed to load agent {agentname} from {module_path}: {e}")
        except AttributeError as e:
            logger.error(f"❌ 属性错误: {e}")
            raise AttributeError(f"Failed to load agent {agentname} from {module_path}: {e}")
        except Exception as e:
            logger.error(f"❌ 未知错误: {e}")
            raise Exception(f"Failed to load agent {agentname} from {module_path}: {e}")

    def try_create_artifact_service(self, ak: str = None, app_key: str = None, project_id: int = None):
        """Try to import and call create_artifact_service(ak, app_key, project_id) from agent module.

        Returns the artifact service instance if available; otherwise returns None.
        """
        agentcfg = self.config.get("agent", {})
        module_path = agentcfg.get("module", "agent.subagent")

        try:
            # Resolve module same as get_agent
            if '/' in module_path or '\\' in module_path or module_path.endswith('.py'):
                file_path = Path(module_path)
                if not file_path.is_absolute():
                    user_working_dir = os.environ.get('USER_WORKING_DIR', os.getcwd())
                    file_path = Path(user_working_dir) / file_path
                if not file_path.exists():
                    logger.warning(f"artifact_service module file not found: {file_path}")
                    return None
                module_name = str(file_path).replace('/', '_').replace('\\', '_').replace('.py', '').replace('.', '_')
                module_name = 'agent_' + module_name.strip('_')
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                if spec is None or spec.loader is None:
                    logger.warning(f"Cannot load module from file for artifact_service: {file_path}")
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(module_path)

            if hasattr(module, 'create_artifact_service'):
                import inspect
                sig = inspect.signature(module.create_artifact_service)
                params = {}
                if 'ak' in sig.parameters:
                    params['ak'] = ak
                if 'app_key' in sig.parameters:
                    params['app_key'] = app_key
                if 'project_id' in sig.parameters:
                    params['project_id'] = project_id
                svc = module.create_artifact_service(**params)
                logger.info("✅ 使用 agent.create_artifact_service 创建 artifact_service 成功")
                return svc
            else:
                logger.info("ℹ️ agent 模块未提供 create_artifact_service，跳过设置 artifact_service")
                return None
        except Exception as e:
            logger.warning(f"⚠️ 尝试创建 artifact_service 失败: {e}")
            return None
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-specific configuration"""
        return self.config.get("ui", {})
    
    def get_files_config(self) -> Dict[str, Any]:
        """Get file handling configuration"""
        return self.config.get("files", {})
    
    def get_websocket_config(self) -> Dict[str, Any]:
        """Get WebSocket configuration"""
        return self.config.get("websocket", {})
    
    def get_tool_display_name(self, tool_name: str) -> str:
        """Get display name for a tool"""
        tools_config = self.config.get("tools", {})
        display_names = tools_config.get("displayNames", {})
        return display_names.get(tool_name, tool_name)
    
    def is_long_running_tool(self, tool_name: str) -> bool:
        """Check if a tool is marked as long-running"""
        tools_config = self.config.get("tools", {})
        long_running = tools_config.get("longRunningTools", [])
        return tool_name in long_running
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration including port and allowed hosts"""
        # 默认主机始终被允许
        default_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        
        server_config = self.config.get("server", {})
        
        # 支持 "host" 字段（用户配置）和 "allowedHosts"（向后兼容）
        user_hosts = server_config.get("host", server_config.get("allowedHosts", []))
        
        # 确保 user_hosts 是列表
        if isinstance(user_hosts, str):
            user_hosts = [user_hosts]
        elif not isinstance(user_hosts, list):
            user_hosts = []
        
        # 如果用户配置了 "*"，则允许所有主机
        if "*" in user_hosts:
            all_hosts = ["*"]
        else:
            # 合并默认主机和用户定义的额外主机
            all_hosts = list(set(default_hosts + user_hosts))  # 使用 set 去重
        
        # 使用环境变量或配置的端口，最后使用默认端口
        default_port = int(os.environ.get('AGENT_SERVER_PORT', '50002'))
        port = server_config.get("port", default_port)
        
        return {
            "port": port,
            "allowedHosts": all_hosts
        }

# Singleton instance
agentconfig = AgentConfig()