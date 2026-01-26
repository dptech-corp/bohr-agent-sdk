"""
Session manager - using ADK native DatabaseSessionService implementation
"""
import os
import json
import uuid
import asyncio
import traceback
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
from fastapi import WebSocket
from google.adk import Runner
from google.genai import types
from google.adk.sessions import DatabaseSessionService, InMemorySessionService, Session

from server.connection import ConnectionContext
from server.user_files import UserFileManager
from config.agent_config import agentconfig

# Configure logging output to file
# Use relative path to project root or environment variable configuration
log_file_path = os.environ.get('WEBSOCKET_LOG_PATH', './websocket.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Session manager
    Based on ADK native DatabaseSessionService, providing session management, persistence, user isolation
    """
    
    # Constants
    MAX_WAIT_TIME = 5  # Max wait time for runner initialization (seconds)
    WAIT_INTERVAL = 0.1  # Wait interval (seconds)
    MAX_CONTEXT_MESSAGES = 8  # Max messages in context
    
    def __init__(self):
        """Initialize session manager"""
        # Active connection management
        self.active_connections: Dict[WebSocket, ConnectionContext] = {}
        
        # Application config
        self.app_name = agentconfig.config.get("agent", {}).get("name", "Agent")
        
        # Initialize paths
        user_working_dir = os.environ.get('USER_WORKING_DIR', os.getcwd())
        files_config = agentconfig.get_files_config()
        sessions_dir = files_config.get('sessionsDir', '.agent_sessions')
        
        # Session storage directory
        sessions_path = Path(sessions_dir)
        if sessions_path.is_absolute():
            self.sessions_dir = sessions_path
        else:
            self.sessions_dir = Path(user_working_dir) / sessions_dir
            
        # Ensure directory exists
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # SessionService cache (independent instance for each user)
        self.session_services: Dict[str, Any] = {}
        
        # Initialize user file manager
        self.user_file_manager = UserFileManager(user_working_dir, str(self.sessions_dir))
        
        # Runner cache
        self.runners: Dict[str, Runner] = {}
        
        # Runner error cache
        self._runner_errors: Dict[str, str] = {}
        
    def _create_session_service(self, user_identifier: str, is_registered: bool):
        """
        Create SessionService for user
        
        Args:
            user_identifier: User identifier
            is_registered: Whether user is registered
            
        Returns:
            SessionService instance
        """
        if is_registered:
            # Registered users use DatabaseSessionService for persistence
            user_db_dir = self.sessions_dir / "users" / user_identifier
            user_db_dir.mkdir(parents=True, exist_ok=True)
            
            db_file = user_db_dir / "sessions.db"
            db_url = f"sqlite:///{db_file}"
            
            return DatabaseSessionService(db_url=db_url)
        else:
            # Temporary users use in-memory storage
            return InMemorySessionService()
            
    async def connect_client(self, websocket: WebSocket, access_key: str = "", app_key: str = ""):
        """
        Connect new client
        
        Args:
            websocket: WebSocket connection
            access_key: Bohrium access key
            app_key: Bohrium app key
        """
        await websocket.accept()
        
        # Create connection context
        context = ConnectionContext(websocket, access_key, app_key)
        self.active_connections[websocket] = context
        
        # Asynchronously initialize user info
        await context.init_bohrium_user_id()
        user_identifier = context.get_user_identifier()
        is_registered = context.is_registered_user()
        
        # Create independent SessionService for this user
        session_service = self._create_session_service(user_identifier, is_registered)
        self.session_services[user_identifier] = session_service
        
        # Load or create sessions
        await self._load_or_create_sessions(context, session_service)
        
        # Send initial data
        await self._send_initial_data(context, session_service)
        
    async def disconnect_client(self, websocket: WebSocket):
        """
        Disconnect client
        
        Args:
            websocket: WebSocket connection
        """
        if websocket not in self.active_connections:
            return
            
        context = self.active_connections[websocket]
        user_identifier = context.get_user_identifier()
        
        # Mark connection as disconnected to prevent further operations
        context.is_connected = False
        
        # Clean up SessionService
        if user_identifier in self.session_services:
            del self.session_services[user_identifier]
            
        # Clean up Runner
        for key in list(self.runners.keys()):
            if key.startswith(f"{user_identifier}_"):
                del self.runners[key]
        
        # Clean up related error cache
        for key in list(self._runner_errors.keys()):
            if key.startswith(f"{user_identifier}_"):
                del self._runner_errors[key]
                
        # Clean up connection context
        context.cleanup()
        del self.active_connections[websocket]
        
    async def create_session(self, context: ConnectionContext) -> Session:
        """
        Create new session
        
        Args:
            context: Connection context
            
        Returns:
            Session ID
        """
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        
        if not session_service:
            return None
            
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "last_message_at": datetime.now().isoformat(),
            "message_count": 0,
            "title": "Untitled",
            "project_id": context.project_id
        }
        
        session = await session_service.create_session(
            app_name=self.app_name,
            user_id=user_identifier,
            session_id=session_id,
            state={"metadata": metadata}
        )
        
        # Asynchronously initialize Runner
        asyncio.create_task(self._init_runner(context, session.id))
        
        # Update current session
        context.current_session_id = session.id
        
        return session
        
    async def delete_session(self, context: ConnectionContext, session_id: str) -> bool:
        """
        Delete session
        
        Args:
            context: Connection context
            session_id: Session ID
            
        Returns:
            Whether deletion succeeded
        """
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        
        if not session_service:
            return False
            
        try:
            # delete_session method returns None, not boolean
            await session_service.delete_session(
                app_name=self.app_name,
                user_id=user_identifier,
                session_id=session_id
            )
            
            # No exception means success
            
            # Clean up Runner
            runner_key = f"{user_identifier}_{session_id}"
            if runner_key in self.runners:
                del self.runners[runner_key]
                
            return True  # Return success if no exception
        except Exception as e:
            return False
        
    async def switch_session(self, context: ConnectionContext, session_id: str) -> bool:
        """
        Switch current session
        
        Args:
            context: Connection context
            session_id: Session ID
            
        Returns:
            Whether switch succeeded
        """
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        
        if not session_service:
            return False
            
        # Check if session exists
        session = await session_service.get_session(
            app_name=self.app_name,
            user_id=user_identifier,
            session_id=session_id
        )
        
        if not session:
            return False
            
        # Switch session
        context.current_session_id = session_id
        
        # Ensure Runner is initialized
        runner_key = f"{user_identifier}_{session_id}"
        if runner_key not in self.runners:
            asyncio.create_task(self._init_runner(context, session_id))
            
        return True
        
    async def process_message(self, context: ConnectionContext, message: str, attachments: list = None):
        # Save context reference for URL generation
        self.current_context = context
        """
        Process user message
        
        Args:
            context: Connection context
            message: User message
        """
        # Check project_id
        if not context.project_id and not os.environ.get('BOHR_PROJECT_ID'):
            await self._send_error(context, "🔒 请先设置项目 ID")
            return
            
        if not context.current_session_id:
            await self._send_error(context, "没有活动的会话")
            return
            
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        
        if not session_service:
            await self._send_error(context, "会话服务未初始化")
            return
            
        # 获取会话
        session = await session_service.get_session(
            app_name=self.app_name,
            user_id=user_identifier,
            session_id=context.current_session_id
        )
        
        if not session:
            await self._send_error(context, "会话不存在")
            return
            
        # 等待 Runner 初始化
        runner = await self._get_or_wait_runner(context, context.current_session_id)
        if not runner:
            error_details = self._runner_errors.get(f"{user_identifier}_{context.current_session_id}", "未知错误")
            await self._send_error(
                context, 
                f"会话初始化失败\n\n可能的原因：\n"
                f"1. Agent 配置文件路径错误\n"
                f"2. Agent 模块导入失败\n"
                f"3. Project ID 无效\n\n"
                f"错误详情：{error_details}\n\n"
                f"请检查 config/agent-config.json 中的配置"
            )
            return
            
        # 更新会话元数据（在处理消息之前）
        await self._update_session_metadata(context, session, message)
        
        # Get user file directory
        user_files_dir = self.user_file_manager.get_user_files_dir(user_id=user_identifier)
        original_cwd = os.getcwd()
        
        try:
            # Switch to user file directory
            os.chdir(user_files_dir)
            
            # Build message content
            content = self._build_message_content(session, message, attachments)
            
            # Process message stream
            await self._process_message_stream(
                context,
                runner,
                content,
                user_identifier,
                context.current_session_id
            )
            
        except Exception as e:
            await self._send_error(context, f"处理消息失败: {str(e)}")
            
        finally:
            # Restore working directory
            try:
                os.chdir(original_cwd)
            except Exception as e:
                pass
                
    async def _load_or_create_sessions(self, context: ConnectionContext, session_service):
        """Load or create sessions"""
        user_identifier = context.get_user_identifier()
        
        try:
            response = await session_service.list_sessions(
                app_name=self.app_name,
                user_id=user_identifier
            )
            
            # Get session list from ListSessionsResponse object
            sessions = response.sessions if hasattr(response, 'sessions') else []
            
            if sessions:
                # Sort by last message time
                sessions.sort(
                    key=lambda s: self._get_session_last_update_time(s),
                    reverse=True
                )
                
                # Select most recent session as current
                context.current_session_id = sessions[0].id
                
                # Initialize Runner for each session
                for session in sessions:
                    asyncio.create_task(self._init_runner(context, session.id))
                    
            else:
                # Create new session
                await self._create_default_session(context, session_service)
                
        except Exception as e:
            await self._create_default_session(context, session_service)
            
    async def _create_default_session(self, context: ConnectionContext, session_service):
        """Create default session"""
        user_identifier = context.get_user_identifier()
        session_id = str(uuid.uuid4())
        
        # Create session metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "last_message_at": datetime.now().isoformat(),
            "message_count": 0,
            "title": "Untitled",
            "project_id": context.project_id
        }
        
        session = await session_service.create_session(
            app_name=self.app_name,
            user_id=user_identifier,
            session_id=session_id,
            state={"metadata": metadata}
        )
        
        context.current_session_id = session.id
        
        # Initialize Runner
        asyncio.create_task(self._init_runner(context, session.id))
        
    async def _init_runner(self, context: ConnectionContext, session_id: str, retry_count: int = 0):
        """Asynchronously initialize Runner with retry mechanism"""
        user_identifier = context.get_user_identifier()
        runner_key = f"{user_identifier}_{session_id}"
        max_retries = 3
        
        logger.info(f"🚀 开始初始化 Runner: {runner_key} (尝试 {retry_count + 1}/{max_retries})")
        logger.debug(f"  用户标识: {user_identifier}")
        logger.debug(f"  会话ID: {session_id}")
        logger.debug(f"  Access Key: {'有' if context.access_key else '无'}")
        logger.debug(f"  App Key: {'有' if context.app_key else '无'}")
        
        try:
            # Get project_id
            project_id = context.project_id or os.environ.get('BOHR_PROJECT_ID')
            if project_id:
                project_id = int(project_id) if isinstance(project_id, str) else project_id
            logger.debug(f"  Project ID: {project_id}")
                
            # Create agent
            logger.info(f"📦 创建 Agent...")
            logger.debug(f"  配置模块: {agentconfig.config.get('agent', {}).get('module')}")
            logger.debug(f"  Agent名称: {agentconfig.config.get('agent', {}).get('name')}")
            
            loop = asyncio.get_event_loop()
            user_agent = await loop.run_in_executor(
                None,
                agentconfig.get_agent,
                context.access_key or "",
                context.app_key or "",
                project_id
            )
            
            if not user_agent:
                raise ValueError("Agent 创建失败: 返回 None")
            
            logger.info(f"✅ Agent 创建成功: {type(user_agent).__name__}")
            
            # Create Runner
            logger.info(f"🏃 创建 Runner...")
            
            # 检查连接是否仍然有效
            if not context.is_connected:
                logger.warning(f"⚠️ 连接已断开，跳过 Runner 初始化: {runner_key}")
                return
                
            session_service = self.session_services.get(user_identifier)
            if not session_service:
                logger.warning(f"⚠️ SessionService 已被清理，跳过 Runner 初始化: {runner_key}")
                return
                
            runner = Runner(
                agent=user_agent,
                session_service=session_service,
                app_name=self.app_name
            )
            
            self.runners[runner_key] = runner
            logger.info(f"✅ Runner 初始化成功: {runner_key}")
            logger.debug(f"  当前 Runner 数量: {len(self.runners)}")
            
            # 清除之前的错误记录
            if runner_key in self._runner_errors:
                del self._runner_errors[runner_key]
            
        except (ImportError, Exception) as e:
            error_type = "导入错误" if isinstance(e, ImportError) else "Runner 初始化失败"
            error_msg = f"❌ {error_type}: {str(e)}\n类型: {type(e).__name__}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # 如果还有重试机会
            if retry_count < max_retries - 1:
                logger.info(f"🔄 准备重试 Runner 初始化: {runner_key}")
                # 清理可能的部分初始化状态
                if runner_key in self.runners:
                    del self.runners[runner_key]
                
                # 等待一小段时间后重试
                await asyncio.sleep(1)
                
                # 递归调用自己进行重试
                await self._init_runner(context, session_id, retry_count + 1)
            else:
                # 所有重试都失败，存储错误信息
                self._runner_errors[runner_key] = f"{error_msg}\n\n已尝试 {max_retries} 次初始化，全部失败。"
                logger.error(f"❌ Runner 初始化彻底失败: {runner_key}，已尝试 {max_retries} 次")
            
    async def _get_or_wait_runner(self, context: ConnectionContext, session_id: str) -> Optional[Runner]:
        """Get or wait for Runner initialization with auto-recovery"""
        user_identifier = context.get_user_identifier()
        runner_key = f"{user_identifier}_{session_id}"
        
        logger.debug(f"⏳ 等待 Runner 初始化: {runner_key}")
        
        # Wait for Runner initialization
        retry_count = 0
        max_retries = int(self.MAX_WAIT_TIME / self.WAIT_INTERVAL)
        recovery_attempted = False
        
        while runner_key not in self.runners and retry_count < max_retries:
            # 检查是否有错误
            if runner_key in self._runner_errors:
                logger.error(f"Runner 初始化已失败: {self._runner_errors[runner_key]}")
                
                # 如果还没有尝试过恢复，尝试一次
                if not recovery_attempted:
                    recovery_attempted = True
                    logger.info(f"🔧 尝试自动恢复 Runner: {runner_key}")
                    
                    # 清除错误记录
                    del self._runner_errors[runner_key]
                    
                    # 触发新的初始化尝试
                    asyncio.create_task(self._init_runner(context, session_id))
                    
                    # 继续等待
                    await asyncio.sleep(self.WAIT_INTERVAL)
                    retry_count += 1
                    continue
                else:
                    # 已经尝试过恢复但仍然失败
                    # 发送详细的错误信息到前端
                    await self._send_error(
                        context, 
                        f"会话初始化失败\n\n错误详情：\n{self._runner_errors.get(runner_key, '未知错误')}\n\n"
                        f"建议：\n"
                        f"1. 请尝试新建一个会话\n"
                        f"2. 检查 Agent 配置是否正确\n"
                        f"3. 确认 Project ID 是否有效"
                    )
                    # 清除错误缓存
                    if runner_key in self._runner_errors:
                        del self._runner_errors[runner_key]
                    return None
                
            await asyncio.sleep(self.WAIT_INTERVAL)
            retry_count += 1
            
            if retry_count % 10 == 0:  # 每秒记录一次
                logger.debug(f"  仍在等待... ({retry_count * self.WAIT_INTERVAL:.1f}秒)")
        
        runner = self.runners.get(runner_key)
        if runner:
            logger.info(f"✅ 获取 Runner 成功: {runner_key}")
        else:
            logger.error(f"❌ 超时等待 Runner: {runner_key} (等待了 {self.MAX_WAIT_TIME} 秒)")
            # 如果超时且没有错误记录，可能是初始化太慢
            if runner_key not in self._runner_errors:
                await self._send_error(
                    context,
                    f"会话初始化超时\n\n"
                    f"可能的原因：\n"
                    f"1. Agent 初始化时间过长\n"
                    f"2. 系统资源不足\n\n"
                    f"建议尝试新建会话"
                )
            
        return runner
        
    async def _update_session_metadata(self, context: ConnectionContext, session: Session, message: str):
        """Correctly update metadata in session.state through append_event"""
        # Get existing metadata
        metadata = session.state.get('metadata', {}) if session.state else {}
        
        # Prepare new metadata
        new_metadata = dict(metadata)  # Create copy
        new_metadata['last_message_at'] = datetime.now().isoformat()
        new_metadata['message_count'] = new_metadata.get('message_count', 0) + 1
        
        # Use message content as title for first message
        if new_metadata['message_count'] == 1:
            new_metadata['title'] = message[:50] if len(message) > 50 else message
            
        # Create state_delta through EventActions
        from google.adk.events import Event, EventActions
        
        state_delta = {
            'metadata': new_metadata
        }
        
        # Create event containing state_delta
        update_event = Event(
            invocation_id=f"metadata_update_{datetime.now().timestamp()}",
            author="system",
            actions=EventActions(state_delta=state_delta),
            timestamp=datetime.now().timestamp()
        )
        
        # Update state correctly through append_event
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        if session_service:
            await session_service.append_event(session, update_event)
        
    async def _process_message_stream(
        self,
        context: ConnectionContext,
        runner: Runner,
        content: types.Content,
        user_identifier: str,
        session_id: str
    ):
        """Process message stream - using ADK native event handling"""
        streaming_text = ""  # Accumulate streaming text

        # Run Runner
        async for event in runner.run_async(
            new_message=content,
            user_id=user_identifier,
            session_id=session_id
        ):
            # 1. Check event author
            author = getattr(event, 'author', None)
            
            # 2. Check if it's streaming output
            is_partial = getattr(event, 'partial', False)
            
            # 3. Handle function calls (tool call requests)
            function_calls = event.get_function_calls() if hasattr(event, 'get_function_calls') else []
            if function_calls:
                for call in function_calls:
                    await self._send_message(context, {
                        "type": "tool",
                        "tool_name": call.name,
                        "args": call.args,  # Add tool call parameters
                        "status": "executing",
                        "timestamp": datetime.now().isoformat()
                    })
                    await asyncio.sleep(0.2)
            # 4. Handle function responses (tool execution results)
            function_responses = event.get_function_responses() if hasattr(event, 'get_function_responses') else []
            if function_responses:
                for response in function_responses:
                    # Format response result
                    result_str = self._format_response_data(response.response)
                    await self._send_message(context, {
                        "type": "tool",
                        "tool_name": response.name,
                        "result": result_str,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # 5. Handle text content
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            if is_partial:
                                # Streaming text, accumulate
                                streaming_text += part.text
                            else:
                                # Complete text
                                text_to_send = streaming_text + part.text if streaming_text else part.text
                                streaming_text = ""  # Reset accumulator
                                
                                # Send message based on role
                                role = getattr(event.content, 'role', 'model')
                                if role == 'model':
                                    await self._send_message(context, {
                                        "type": "assistant",
                                        "content": text_to_send,
                                        "session_id": session_id
                                    })
            
            # 6. Check if it's final response
            if hasattr(event, 'is_final_response') and event.is_final_response():
                # If there's accumulated streaming text, send it now
                if streaming_text:
                    await self._send_message(context, {
                        "type": "assistant",
                        "content": streaming_text,
                        "session_id": session_id
                    })
                    streaming_text = ""
            
            # 7. Handle Actions (state changes and control flow)
            if hasattr(event, 'actions') and event.actions:
                # State changes
                if hasattr(event.actions, 'state_delta') and event.actions.state_delta:
                    pass
                
                # Skip summarization flag
                if hasattr(event.actions, 'skip_summarization') and event.actions.skip_summarization:
                    pass
                
                # Agent transfer
                if hasattr(event.actions, 'transfer_to_agent') and event.actions.transfer_to_agent:
                    pass
        
        # Send completion marker
        await self._send_message(context, {
            "type": "complete",
            "content": ""
        })
        
        # Send updated session list
        await self.send_sessions_list(context)
        
    # _handle_tool_events method removed, functionality integrated into _process_message_stream
                    
    async def _get_session_metadata(self, session_service, user_identifier: str, session_id: str) -> dict:
        """Get latest session metadata"""
        try:
            fresh_session = await session_service.get_session(
                app_name=self.app_name,
                user_id=user_identifier,
                session_id=session_id
            )
            if fresh_session and hasattr(fresh_session, 'state') and isinstance(fresh_session.state, dict):
                return fresh_session.state.get('metadata', {})
        except Exception as e:
            pass
        return {}
    
    async def _send_initial_data(self, context: ConnectionContext, session_service):
        """Send initial data to client"""
        user_identifier = context.get_user_identifier()
        
        # Send session list
        response = await session_service.list_sessions(
            app_name=self.app_name,
            user_id=user_identifier
        )
        
        # Get session list from ListSessionsResponse object
        sessions = response.sessions if hasattr(response, 'sessions') else []
        
        sessions_data = []
        for session in sessions:
            # Get latest metadata
            metadata = await self._get_session_metadata(session_service, user_identifier, session.id)
            if not metadata:  # If fetch fails, use original data as fallback
                metadata = session.state.get('metadata', {}) if session.state else {}
                
            sessions_data.append({
                "id": session.id,
                "title": metadata.get("title", "Untitled"),
                "created_at": metadata.get("created_at", datetime.now().isoformat()),
                "last_message_at": metadata.get("last_message_at", datetime.now().isoformat()),
                "message_count": metadata.get("message_count", 0)
            })
            
        await self._send_message(context, {
            "type": "sessions_list",
            "sessions": sessions_data,
            "current_session_id": context.current_session_id
        })
        
        # Send current session message history
        if context.current_session_id:
            await self._send_session_messages(context, session_service, context.current_session_id)
            
    async def _send_session_messages(self, context: ConnectionContext, session_service, session_id: str):
        """Send session message history"""
        user_identifier = context.get_user_identifier()
        
        session = await session_service.get_session(
            app_name=self.app_name,
            user_id=user_identifier,
            session_id=session_id
        )
        
        if not session or not hasattr(session, 'events'):
            return
            
        messages_data = []
        
        for event in session.events:
            # Parse events, convert to frontend-understandable format
            if not hasattr(event, 'content'):
                continue
                
            content = event.content
            role = getattr(content, 'role', None)
            timestamp = self._format_timestamp(getattr(event, "timestamp", None))
            
            # Handle message content
            if hasattr(content, 'parts'):
                for part in content.parts:
                    # Handle text messages
                    if hasattr(part, 'text') and part.text:
                        if role == 'user':
                            messages_data.append({
                                "id": str(uuid.uuid4()),
                                "role": "user",
                                "type": "user",
                                "content": part.text,
                                "timestamp": timestamp
                            })
                        elif role == 'model':
                            messages_data.append({
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "type": "assistant",
                                "content": part.text,
                                "timestamp": timestamp
                            })
                    
                    # Handle tool calls - don't show executing state in history
                    elif hasattr(part, 'function_call') and part.function_call:
                        # Skip function_call in history, only show final results
                        pass
                    
                    # Handle tool responses - only show completed tool calls
                    elif hasattr(part, 'function_response') and part.function_response:
                        func_resp = part.function_response
                        tool_name = getattr(func_resp, 'name', 'unknown')
                        result_str = self._format_response_data(getattr(func_resp, 'response', {}))
                        # Use simple UUID for history messages
                        messages_data.append({
                            "id": str(uuid.uuid4()),
                            "role": "tool",
                            "type": "tool",
                            "tool_name": tool_name,
                            "tool_status": "completed",
                            "content": result_str,
                            "timestamp": timestamp
                        })
            else:
                # Simple text content
                if role == 'user':
                    messages_data.append({
                        "id": str(uuid.uuid4()),
                        "role": "user",
                        "type": "user",
                        "content": str(content),
                        "timestamp": timestamp
                    })
                elif role == 'model':
                    messages_data.append({
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "type": "assistant",
                        "content": str(content),
                        "timestamp": timestamp
                    })
                
        await self._send_message(context, {
            "type": "session_messages",
            "session_id": session_id,
            "messages": messages_data
        })
        
    async def _send_message(self, context: ConnectionContext, message: dict):
        """Send message to client"""
        if 'id' not in message:
            message['id'] = f"{message.get('type', 'unknown')}_{datetime.now().timestamp()}"
            
        try:
            await context.websocket.send_json(message)
        except Exception as e:
            asyncio.create_task(self.disconnect_client(context.websocket))
            
    async def _send_error(self, context: ConnectionContext, error_message: str):
        """Send error message"""
        await self._send_message(context, {
            "type": "error",
            "content": error_message
        })
        
    def _get_session_last_update_time(self, session: Session) -> datetime:
        """Get session last update time"""
        # Get metadata from session.state
        if hasattr(session, 'state') and isinstance(session.state, dict):
            metadata = session.state.get('metadata', {})
            last_message_at = metadata.get('last_message_at')
            if last_message_at:
                try:
                    return datetime.fromisoformat(last_message_at)
                except:
                    pass
                    
        # Use ADK native last_update_time
        if hasattr(session, 'last_update_time'):
            return datetime.fromtimestamp(session.last_update_time)
            
        return datetime.min
        
    def _get_base_url(self, context: ConnectionContext) -> str:
        """动态获取基础URL"""
        headers = getattr(context, 'request_headers', {})
        
        # 1. 从Origin头获取
        origin = headers.get('origin', '')
        if origin:
            return origin
        
        # 2. 从Host头获取
        host = headers.get('host', '')
        if host:
            forwarded_proto = headers.get('x-forwarded-proto', '')
            protocol = 'https' if forwarded_proto == 'https' else 'http'
            return f"{protocol}://{host}"
        
        # 3. 从环境变量获取
        base_url = os.environ.get('AGENT_API_URL', '')
        if base_url:
            return base_url.rstrip('/')
        
        # 4. 默认值
        return "http://localhost:8000"

    def _build_message_content(self, session, message: str, attachments: list = None) -> types.Content:
        """Build message content (including history context and attachments)"""
        # Build message with file attachment information
        enhanced_message = message
        
        if attachments and hasattr(self, 'current_context'):
            # 获取基础URL和用户ID
            base_url = self._get_base_url(self.current_context)
            user_id = self.current_context.get_user_identifier()
            
            file_info = "\n\n用户已上传文件，请你跟据上传的文件路径，调用传入所希望调用的工具中，："
            for att in attachments:
                file_info += f"\n  文件路径: {att['relative_path']}"
                
            enhanced_message = message + file_info if message else file_info.strip()
        
        return types.Content(
            role='user',
            parts=[types.Part(text=enhanced_message)]
        )
        
    def _format_response_data(self, response_data):
        """Format response data"""
        if isinstance(response_data, (dict, list, tuple)):
            try:
                return json.dumps(response_data, indent=2, ensure_ascii=False)
            except:
                return str(response_data)
        return str(response_data) if not isinstance(response_data, str) else response_data
        
    def _extract_final_response(self, events: list) -> Optional[str]:
        """Extract final response from event list"""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                content = event.content
                if hasattr(content, 'parts') and content.parts:
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return '\n'.join(text_parts)
        return None
        
    def _event_to_message_data(self, event) -> Optional[dict]:
        """Convert event to message data"""
        if not event:
            return None
            
        # Handle different types of events
        message_data = {
            "id": str(uuid.uuid4()),
            "timestamp": self._format_timestamp(getattr(event, "timestamp", None))
        }
        
        # Extract info based on event type
        if hasattr(event, 'type'):
            message_data["type"] = event.type
            
        if hasattr(event, 'role'):
            # Unify role field: convert role to frontend-expected type format
            role = event.role
            if role == 'model':
                message_data["type"] = "assistant"
            elif role == 'user':
                message_data["type"] = "user"
            else:
                message_data["type"] = role
            message_data["role"] = role  # Preserve original role info
            
        if hasattr(event, 'content'):
            # Handle Content objects
            content = event.content
            if hasattr(content, 'parts'):
                text_parts = []
                tool_calls = []
                tool_responses = []
                
                for part in content.parts:
                    # Handle text content
                    if hasattr(part, 'text') and part.text is not None:
                        text_parts.append(part.text)
                    
                    # Handle tool calls
                    if hasattr(part, 'function_call'):
                        func_call = part.function_call
                        tool_calls.append({
                            "id": getattr(func_call, 'id', ''),
                            "name": getattr(func_call, 'name', ''),
                            "args": getattr(func_call, 'args', {})
                        })
                    
                    # Handle tool responses
                    if hasattr(part, 'function_response'):
                        func_resp = part.function_response
                        tool_responses.append({
                            "id": getattr(func_resp, 'id', ''),
                            "name": getattr(func_resp, 'name', ''),
                            "response": getattr(func_resp, 'response', {})
                        })
                
                # Set message content
                if text_parts:
                    message_data["content"] = '\n'.join(text_parts)
                
                # Set tool call info
                if tool_calls:
                    message_data["tool_calls"] = tool_calls
                
                if tool_responses:
                    message_data["tool_responses"] = tool_responses
                    
            else:
                message_data["content"] = str(content)
        
        # Only return messages with content
        if "content" in message_data or "tool_calls" in message_data or "tool_responses" in message_data:
            return message_data
            
        return None
        
    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp"""
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()
        
        if isinstance(timestamp, (int, float)):
            # Convert Unix timestamp to ISO format
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        
        if isinstance(timestamp, str):
            return timestamp
            
        return datetime.now(timezone.utc).isoformat()
        
        
    def get_user_identifier_from_request(self, access_key: str = None, app_key: str = None) -> Optional[str]:
        """
        Get user identifier from request info (prefer from connected context)
        
        Args:
            access_key: Bohrium access key
            app_key: Bohrium app key (reserved for future extension)
            
        Returns:
            User identifier or None
        """
        if access_key:
            # Check if there's a connected user
            for context in self.active_connections.values():
                if context.access_key == access_key:
                    return context.get_user_identifier()
        return None
        
    async def send_sessions_list(self, context: ConnectionContext):
        """
        Send session list to client
        
        Args:
            context: Connection context
        """
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        
        if not session_service:
            await self._send_error(context, "会话服务未初始化")
            return
            
        try:
            # Get session list
            response = await session_service.list_sessions(
                app_name=self.app_name,
                user_id=user_identifier
            )
            
            # Get session list from ListSessionsResponse object
            sessions = response.sessions if hasattr(response, 'sessions') else []
            
            sessions_data = []
            for session in sessions:
                # Uniformly use helper method to get latest metadata
                metadata = await self._get_session_metadata(session_service, user_identifier, session.id)
                if not metadata:  # If fetch fails, use original data as fallback
                    metadata = session.state.get('metadata', {}) if session.state else {}
                
                title = metadata.get("title", "Untitled")
                    
                sessions_data.append({
                    "id": session.id,
                    "title": title,
                    "created_at": metadata.get("created_at", datetime.now().isoformat()),
                    "last_message_at": metadata.get("last_message_at", datetime.now().isoformat()),
                    "message_count": metadata.get("message_count", 0)
                })
                
            await self._send_message(context, {
                "type": "sessions_list",
                "sessions": sessions_data,
                "current_session_id": context.current_session_id
            })
            
        except Exception as e:
            await self._send_error(context, "获取会话列表失败")
            
    async def send_session_messages(self, context: ConnectionContext, session_id: str):
        """
        Send message history for specified session
        
        Args:
            context: Connection context
            session_id: Session ID
        """
        user_identifier = context.get_user_identifier()
        session_service = self.session_services.get(user_identifier)
        
        if not session_service:
            await self._send_error(context, "会话服务未初始化")
            return
            
        try:
            # Directly call internal method, reuse logic
            await self._send_session_messages(context, session_service, session_id)
            
        except Exception as e:
            await self._send_error(context, "获取会话消息失败")