#!/usr/bin/env python
# coding=utf-8
import json
import time
import os
import hmac
import base64
from hashlib import sha1
from paho.mqtt import client as mqtt
import logging
import dotenv
from typing import Callable, Union
from pathlib import Path

from .device.device import Device
from .device.types import BaseParams, ActionResult

# Set up logging
logger = logging.getLogger("lab")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Add formatter to ch
    ch.setFormatter(formatter)
    # Add ch to logger
    logger.addHandler(ch)


class DeviceTwin:
    """Device Twin class for handling MQTT communication with devices.
    
    This class provides methods to initialize and run a device twin that can
    receive control commands and publish status updates via MQTT.
    """
    
    def __init__(self, device: Union[Device, Callable[[str, str, BaseParams], ActionResult]], env_path: str = None):
        """Initialize the Device Twin with a device or a device action dispatcher function.
        
        Args:
            device: Either a Device object or a function to dispatch device actions
            env_path: Optional path to the .env file. If not provided, will try to load from current directory
        """
        self.mqtt_client = None
        
        # Load environment variables
        if env_path:
            # 如果指定了环境变量文件路径，优先加载
            dotenv.load_dotenv(env_path, override=True)
        else:
            # 尝试从当前工作目录加载
            current_dir = os.getcwd()
            env_file = os.path.join(current_dir, '.env')
            if os.path.exists(env_file):
                dotenv.load_dotenv(env_file, override=True)
            else:
                # 如果当前目录没有找到，再尝试加载 SDK 的默认环境变量
                dotenv.load_dotenv()
        
        # Set up device or dispatcher function
        if isinstance(device, Device):
            self.device = device
            self.dispatch_device_actions = device.dispatch_device_actions
        else:
            self.device = None
            self.dispatch_device_actions = device
        
        # MQTT configuration
        self.mqtt_instance_id = os.getenv("MQTT_INSTANCE_ID")
        self.mqtt_endpoint = os.getenv("MQTT_ENDPOINT")
        self.mqtt_device_id = os.getenv("MQTT_DEVICE_ID")
        self.mqtt_group_id = os.getenv("MQTT_GROUP_ID")
        self.mqtt_port = os.getenv("MQTT_PORT")
        self.mqtt_ak = os.getenv("MQTT_AK")
        self.mqtt_sk = os.getenv("MQTT_SK")
        
        # Create client ID from group ID and device ID
        self.client_id = f"{self.mqtt_group_id}@@@{self.mqtt_device_id}_device"
        
        # Topics
        self.device_control_topic = os.getenv("MQTT_DEVICE_CONTROL_TOPIC", "device_control")
        self.device_status_topic = os.getenv("MQTT_DEVICE_STATUS_TOPIC", "device_status")
        
        self._init_mqtt_client()
    
    def _init_mqtt_client(self) -> bool:
        """Initialize the MQTT client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        # Check if all required environment variables are set
        required_vars = ["MQTT_INSTANCE_ID", "MQTT_ENDPOINT", "MQTT_DEVICE_ID", 
                         "MQTT_GROUP_ID", "MQTT_AK", "MQTT_SK"]
        
        missing_vars = [var for var in required_vars if not getattr(self, var.lower(), None)]
        if missing_vars:
            logger.error(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please set these variables in your .env file")
            return False
        
        # Create MQTT client with clean_session=True to avoid session conflicts
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, self.client_id, clean_session=True)
        
        # Set callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_log = self.on_log
        
        # Set up authentication
        username = f'Signature|{self.mqtt_ak}|{self.mqtt_instance_id}'
        password = base64.b64encode(hmac.new(self.mqtt_sk.encode(), self.client_id.encode(), sha1).digest()).decode()
        self.mqtt_client.username_pw_set(username, password)
        
        # Set keep alive interval (higher value for more stable connection)
        self.mqtt_client.keepalive = 120
        
        return True
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the MQTT broker."""
        logger.info(f"Connected to MQTT broker with result code {rc}")
        # Subscribe to device control topic
        client.subscribe(self.device_control_topic)
        logger.info(f"Subscribed to {self.device_control_topic}")
    
    def on_message(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        """Callback for when a message is received from the MQTT broker."""
        try:
            # TODO 过滤掉不是这个设备的消息
            logger.info(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
            
            # Parse the message
            payload = json.loads(msg.payload.decode())
            
            # Extract device information
            device_name = payload.get("device_name")
            device_action = payload.get("device_action")
            device_params = payload.get("device_params", {})
            request_id = payload.get("request_id", "unknown")
            
            if not device_name or not device_action:
                logger.error("Error: Missing device_name or device_action in message")
                return
            
            logger.info(f"Processing request: {device_action} on {device_name}")
            
            # Execute the device action using the provided dispatcher function
            if self.dispatch_device_actions:
                result = self.dispatch_device_actions(device_name, device_action, device_params)
                result_dict = result.to_dict()
                # Prepare status message
                status_message = {
                    "device_name": device_name,
                    "action": device_action,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "result": result_dict
                }
                
                # Publish status update
                client.publish(self.device_status_topic, json.dumps(status_message))
                logger.info(f"Published status update to {self.device_status_topic}")
            else:
                logger.error("No device action dispatcher function provided")
            
        except json.JSONDecodeError:
            logger.error(f"Error: Invalid JSON in message: {msg.payload.decode()}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the MQTT broker."""
        if rc != 0:
            logger.error(f"Unexpected disconnection from MQTT broker: {rc}")
            logger.error("This typically means: authentication issues, network problems, or server-side disconnection", exc_info=True)
    
    def on_log(self, client, userdata, level, buf):
        """Log MQTT client messages for debugging."""
        logger.info(f"MQTT Log: {buf}")
    
    def run(self) -> None:
        """Run the Device Twin MQTT client."""
        if not self.mqtt_client:
            logger.error("MQTT client not initialized.")
            return
        
        # Connect to broker
        logger.info(f"Connecting to MQTT broker at {self.mqtt_endpoint} on port {self.mqtt_port} for device {self.device.device_name}")
        try:
            self.mqtt_client.connect(self.mqtt_endpoint, int(self.mqtt_port), 60)
        
            # Start the loop
            logger.info("Device Twin MQTT client started. Press Ctrl+C to exit.")
            self.mqtt_client.loop_forever()
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {str(e)}")
