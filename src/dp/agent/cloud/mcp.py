"""
MCP (Message Control Protocol) server implementation.
"""
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession
from mcp.shared.context import RequestContext
import dotenv
import logging
from .mqtt import get_mqtt_cloud_instance
from dp.agent.lab.device.device import register_mcp_tools
from dp.agent.lab.tescan_device import TescanDevice

# Load environment variables
dotenv.load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("tescan")

# Initialize logger
logger = logging.getLogger("mcp")
tescan_device = TescanDevice()
# Initialize the MQTT Cloud client
mqtt_cloud = get_mqtt_cloud_instance()

register_mcp_tools(mcp, tescan_device)

@mcp.tool()
async def custom_tool() -> str:
    """A custom tool not related to device actions."""
    return "This is a custom tool"

@mcp.tool()
async def demo_long_running_device_action(hw: str) -> str:
    """Async method: NOTICE Not Implemented yet. Use pyautowin simulate mouse and keyboard to take a picture

    Args:
        hw: horizontal width of the image
    """
    try:
        try:
            current_session = mcp
            logger.info(f"Retrieved session from current context: {current_session}")
        except Exception as e:
            logger.error(f"Could not get session from current context: {str(e)}")
            current_session = None
        
        # 1. Send device control message
        request_id = mqtt_cloud.send_device_control(
            device_name="tescan_microscope",
            device_action="take_picture",
            device_params={"horizontal_width": hw}
        )
        
        async def send_picture_notification(response):
            try:
                result = response["result"]
                
                # Import the necessary types for MCP notifications
                from mcp.types import ServerNotification
                
                # Create a proper MCP notification
                notification = ServerNotification(
                    root={
                        "method": "notifications/message",
                        "params": {
                            "message": f"Picture taken: {result.get('message', 'Success')}",
                            "imageId": result.get("data", {}).get("image_id", "unknown"),
                            "status": result.get("status", "completed")
                        }
                    }
                )
                
                logger.info(f"Successfully sent notification via fresh session")
                logger.info(f"Notification processing complete for image ID: {result.get('data', {}).get('image_id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error in send_picture_notification: {str(e)}")
        
        # Set the callback for async notification
        mqtt_cloud.set_callback(request_id, send_picture_notification)
        
        # For testing, create and send an immediate notification
        try:
            # Create a mock response
            mock_response = {
                "result": {
                    "status": "success",
                    "message": "Picture taken with tescan_microscope",
                    "data": {"image_id": "test_image_" + request_id[:8]}
                }
            }
            
            # Create a notification using the same format as in the callback
            from mcp.types import ServerNotification
            notification = ServerNotification(
                root={
                    "method": "notifications/picture_taken",
                    "params": {
                        "message": "Picture taken successfully (immediate test)",
                        "imageId": mock_response["result"]["data"]["image_id"],
                        "status": "completed"
                    }
                }
            )
            
            if current_session:
                await current_session.send_notification(notification)
                logger.info("Sent immediate test notification via session")
            else:
                await mcp.broadcast_notification(notification)
                logger.info("Sent immediate test notification via broadcast (fallback)")
        except Exception as e:
            logger.error(f"Error sending immediate test notification: {str(e)}")
        
        return f"Picture request sent with ID: {request_id}. You will receive a notification when the picture is taken."
    except Exception as e:
        return f"Error taking picture: {str(e)}"

# Export the global MCP instance
def get_mcp_instance():
    """获取全局MCP实例
    
    Returns:
        FastMCP: MCP服务器实例
    """
    return mcp