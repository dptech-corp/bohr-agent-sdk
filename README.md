# Science Agent SDK

这是DP Tech的Science Agent SDK，提供了一个命令行工具dp-agent，用于管理科学计算任务。同时提供了Python SDK用于开发自定义的科学计算应用。

## 安装

### Linux/Mac

```bash
curl -sSL https://raw.githubusercontent.com/dptech-corp/science-agent-sdk/refs/heads/feat/master/install.sh | bash
```

### Windows

在PowerShell中运行：

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/dptech-corp/science-agent-sdk/refs/heads/feat/master/install.sh -OutFile install.sh
bash install.sh
```

## CLI 使用方法

安装后，您可以使用以下命令：

### 获取资源

```bash
# 获取基础代码结构
dp-agent fetch scaffolding

# 获取配置文件
dp-agent fetch config
```

`fetch config` 命令会下载 .env 配置文件并替换部分动态变量，如 MQTT_DEVICE_ID。
注意：出于安全考虑，此功能仅在内网环境可用。其他环境需要手动配置。

### 运行命令

```bash
# 运行实验环境
dp-agent run lab

# 运行云环境
dp-agent run cloud

# 运行代理
dp-agent run agent

# 调试模式
dp-agent run debug
```

## SDK 快速入门

Science Agent SDK 提供了两种主要的开发模式：实验室模式（Lab）和云模式（Cloud）。

### 基础结构

安装完成并运行 `dp-agent fetch scaffolding` 后，您将获得以下基础项目结构：

```
your-project/
├── lab/                # 实验室模式相关代码
│   ├── __init__.py
│   └── tescan_device.py  # 设备控制示例
├── cloud/              # 云模式相关代码
│   └── __init__.py
└── main.py            # 主程序入口
```

### 实验室模式开发

实验室模式主要用于控制本地实验设备。以下是一个基于 Tescan 设备的示例：

```python
from typing import Dict, TypedDict
from dp.agent.lab.device import Device, action, BaseParams, SuccessResult

class TakePictureParams(BaseParams):
    """拍照参数"""
    horizontal_width: str

class PictureData(TypedDict):
    """照片数据"""
    image_id: str

class PictureResult(SuccessResult):
    """拍照结果"""
    data: PictureData

class MyDevice(Device):
    device_name = "my_device"
    
    @action("take_picture")
    def take_picture(self, params: TakePictureParams) -> PictureResult:
        """拍照动作
        
        Args:
            params: 拍照参数
                - horizontal_width: 图片水平宽度
        """
        hw = params.get("horizontal_width", "default")
        return PictureResult(
            message=f"Picture taken with {self.device_name}",
            data={"image_id": "image_123"}
        )
```

### 云端开发

云模式基于 MCP (Message Control Protocol) 实现，用于处理远程设备控制和任务调度。register_mcp_tools 通过 python 的自省和反射机制实现了设备控制的自动注册，无需重复实现操作定义。
以下展示如何创建设备并注册到 MCP 服务器：

```python
"""
Example of using the science-agent-sdk cloud functionality.
"""
import signal
import sys
from dp.agent.cloud import mcp, get_mqtt_cloud_instance
from dp.agent.lab.device import TescanDevice, register_mcp_tools
from dp.agent.lab.device.device import register_mcp_tools

def signal_handler(sig, frame):
    """Handle SIGINT signal to gracefully shutdown."""
    print("Shutting down...")
    get_mqtt_cloud_instance().stop()
    sys.exit(0)

def main():
    """Start the cloud services."""
    print("Starting Tescan Device Twin Cloud Services...")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create device instance
    device = TescanDevice(mcp, device)
    
    # Register device tools
    register_mcp_tools(device)
    
    # Start MCP server
    print("Starting MCP server...")
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()
```


### 配置说明

在 `.env` 文件中配置必要的环境变量：

```
MQTT_INSTANCE_ID=your_instance_id
MQTT_ENDPOINT=your_endpoint
MQTT_DEVICE_ID=your_device_id
MQTT_GROUP_ID=your_group_id
MQTT_AK=your_access_key
MQTT_SK=your_secret_key
```

### 主要功能

- 设备控制接口（Lab模式）
  - 设备初始化
  - 命令执行
  - 状态监控
  
- 云端任务处理（Cloud模式）
  - 任务队列管理
  - 计算资源调度
  - 结果回传

更详细的API文档请参考代码中的注释。
