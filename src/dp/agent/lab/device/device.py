"""
Device base class.

This module contains the Device base class that users can extend
to implement device-specific functionality.
"""
import inspect
from typing import Dict, Callable, Set, cast, Any, get_type_hints, Type
from functools import wraps
from inspect import Parameter, Signature
from .types import BaseParams, ActionResult
import logging

logger = logging.getLogger("lab")

_ACTION_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {}
_DEVICE_NAME_REGISTRY: Dict[str, str] = {}

def action(device_action: str):
    """Decorator to register a method as a device action and MCP tool.
    
    This decorator captures the method's signature, including parameter types,
    and stores it in the action registry for use by both the device twin
    and the MCP server.
    
    Args:
        device_action: The name of the action
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        logger.info(f"Registering action {device_action} for function {func.__name__}")
        @wraps(func)
        def wrapper(self, params: BaseParams) -> ActionResult:
            return func(self, params)
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        param_info = {}
        for param_name, param in list(sig.parameters.items())[1:]:  # Skip 'self'
            if param_name == 'params' and param.annotation != inspect.Parameter.empty:
                param_class = param.annotation
                if hasattr(param_class, '__annotations__'):
                    for field_name, field_type in param_class.__annotations__.items():
                        param_info[field_name] = {
                            'type': field_type,
                            'required': True,  # Assume required by default
                            'description': f"Parameter: {field_name}"
                        }
        
        return_type = type_hints.get('return', None)
        
        metadata = {
            'func': func,
            'params': param_info,
            'return_type': return_type,
            'doc': func.__doc__
        }
        
        cls_name = None
        
        def register_action(cls):
            logger.info(f"Registering action {device_action} for class {cls.__name__}")
            nonlocal cls_name
            cls_name = cls.__name__
            if cls_name not in _ACTION_REGISTRY:
                _ACTION_REGISTRY[cls_name] = {}
            _ACTION_REGISTRY[cls_name][device_action] = metadata
            return cls
        
        setattr(wrapper, 'register', register_action)
        
        return cast(Callable, wrapper)
    return decorator

class Device:
    """Base class for device implementations.
    
    Users should extend this class and implement device-specific
    functionality using the @action decorator.
    """
    device_name: str = ""
    
    def __init__(self):
        """Initialize the device."""
        self._register_actions()
    
    def _register_actions(self):
        """Register all action methods in the registry."""
        for name, method in inspect.getmembers(self.__class__):
            if hasattr(method, 'register'):
                method.register(self.__class__)
        
        _DEVICE_NAME_REGISTRY[self.__class__.__name__] = self.device_name
    
    def dispatch_device_actions(self, device_name: str, device_action: str, device_params: BaseParams) -> ActionResult:
        """Dispatch a device action.
        
        Args:
            device_name: The name of the device
            device_action: The action to perform
            device_params: Parameters for the action
            
        Returns:
            Result of the action
        """
        if device_name != self.device_name:
            from .types import ErrorResult
            return ErrorResult(f"Unknown device: {device_name}")
        
        cls_name = self.__class__.__name__
        
        if cls_name in _ACTION_REGISTRY and device_action in _ACTION_REGISTRY[cls_name]:
            action_metadata = _ACTION_REGISTRY[cls_name][device_action]
            action_func = action_metadata['func']
            
            try:
                logger.info(f"Executing action {device_action} with params {device_params}")
                return action_func(self, device_params)
            except Exception as e:
                from .types import ErrorResult
                return ErrorResult(f"Error executing action {device_action}: {str(e)}")
        else:
            from .types import ErrorResult
            return ErrorResult(f"Unknown action: {device_action}")
    
    @classmethod
    def get_available_actions(cls) -> Set[str]:
        """Get the set of available actions for this device.
        
        Returns:
            Set of action names
        """
        cls_name = cls.__name__
        if cls_name in _ACTION_REGISTRY:
            return set(_ACTION_REGISTRY[cls_name].keys())
        return set()


def register_mcp_tools(mcp, device: Device):
    """Register actions for the specified device_name as MCP tools.
    
    This function dynamically creates MCP tools for actions that belong to
    the device class that handles the specified device_name.
    
    Args:
        mcp: The MCP server instance
        device: The device to register tools for
    """
    def get_mqtt_instance():
        from dp.agent.cloud import get_mqtt_cloud_instance
        return get_mqtt_cloud_instance()
    
    import logging
    
    logger = logging.getLogger("mcp")
    mqtt_cloud = get_mqtt_instance()
    device_name = device.device_name
    logger.info(f"Registering MCP tools for device_name: {device_name}: {_DEVICE_NAME_REGISTRY}")
    logger.info(f"Available device classes: {_ACTION_REGISTRY.keys()}")
    
    # Find device classes that handle this device_name
    target_cls_names = []
    for cls_name, registered_device_name in _DEVICE_NAME_REGISTRY.items():
        if registered_device_name == device_name:
            target_cls_names.append(cls_name)
            logger.info(f"Found device class {cls_name} for device_name: {device_name}")
    
    if not target_cls_names:
        logger.warning(f"No device class found for device_name: {device_name}")
        logger.info(f"Available device classes: {list(_ACTION_REGISTRY.keys())}")
        logger.info(f"Registered device names: {_DEVICE_NAME_REGISTRY}")
        return
    
    for target_cls_name in target_cls_names:
        if target_cls_name in _ACTION_REGISTRY:
            actions = _ACTION_REGISTRY[target_cls_name]
            logger.info(f"Registering {len(actions)} actions for device class {target_cls_name}")
            
            for action_name, metadata in actions.items():
                # First create the parameter signature for the function
                parameters = []
                for param_name, param_info in metadata['params'].items():
                    parameters.append(
                        Parameter(
                            name=param_name,
                            kind=Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=param_info['type']
                        )
                    )
                
                logger.info(f"Creating MCP tool: {action_name} for device {device_name} with parameters {parameters}")
                
                # Create the function with **kwargs to accept any parameters
                async def create_tool_func(**kwargs):
                    """Dynamically created MCP tool function."""
                    # Get the action_name and metadata from closure
                    nonlocal action_name, metadata
                    
                    params = {}
                    # Extract parameters from kwargs
                    for param_name in metadata['params'].keys():
                        if param_name in kwargs and kwargs[param_name] is not None:
                            params[param_name] = kwargs[param_name]
                    
                    request_id = mqtt_cloud.send_device_control(
                        device_name=device_name,
                        device_action=action_name,
                        device_params=params
                    )
                    
                    response = mqtt_cloud.wait_for_status_update(request_id, timeout=10.0)
                    
                    if response:
                        return str(response["result"])
                    else:
                        return f"Timeout waiting for {action_name} response."
                
                tool_func = create_tool_func
                tool_func.__name__ = action_name
                tool_func.__doc__ = metadata['doc']
                
                new_sig = Signature(
                    parameters=parameters,
                    return_annotation=str  # MCP tools typically return strings
                )
                
                tool_func.__signature__ = new_sig
                
                logger.info(f"Registering MCP tool: {action_name} for device {device_name}")
                mcp.tool()(tool_func)
        else:
            logger.warning(f"No actions found for device class {target_cls_name}")
