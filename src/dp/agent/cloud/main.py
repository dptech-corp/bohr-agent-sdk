#!/usr/bin/env python
# coding=utf-8
"""
Main entry point for the cloud component.

This script starts the MCP server with integrated MQTT client.
"""
import signal
import sys
from dp.agent.cloud import mcp

def signal_handler(sig, frame):
    """Handle SIGINT signal to gracefully shutdown."""
    print("Shutting down...")
    from dp.agent.cloud import get_mqtt_cloud_instance
    get_mqtt_cloud_instance().stop()
    sys.exit(0)

def main():
    """Start the cloud services."""
    print("Starting Tescan Device Twin Cloud Services...")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start MCP server
    print("Starting MCP server...")
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()
