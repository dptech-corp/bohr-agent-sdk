"""
Type definitions for device operations.

This module contains base type definitions for device operations,
including input and output types for device actions.
"""
from typing import Dict, Any, TypedDict

class ActionResult:
    """Base class for all action results."""
    
    def __init__(self, status: str, message: str):
        """Initialize the action result.
        
        Args:
            status: Status of the action ('success' or 'error')
            message: Human-readable message
        """
        self.status = status
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "status": self.status,
            "message": self.message,
            "data": None
        }

class SuccessResult(ActionResult):
    """Success result with data."""
    
    def __init__(self, message: str, data: Any = None):
        """Initialize the success result.
        
        Args:
            message: Human-readable message
            data: Data returned by the action
        """
        super().__init__("success", message)
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "status": self.status,
            "message": self.message,
            "data": self.data
        }

class ErrorResult(ActionResult):
    """Error result."""
    
    def __init__(self, message: str):
        """Initialize the error result.
        
        Args:
            message: Human-readable message
        """
        super().__init__("error", message)

class BaseParams(TypedDict, total=False):
    """Base class for all parameter types."""
    pass
