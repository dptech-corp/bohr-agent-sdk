"""
Tescan device implementation.

This module contains the implementation of a Tescan device with
specific parameter and result types.
"""
import time
from typing import Dict, TypedDict
from dp.agent.lab.device.types import SuccessResult, BaseParams, ErrorResult
from dp.agent.lab.device.device import Device, action

class TakePictureParams(BaseParams):
    """Parameters for take_picture action."""
    horizontal_width: str

class GetStagePositionParams(BaseParams):
    """Parameters for get_stage_position action."""
    pass

class MoveStageParams(BaseParams):
    """Parameters for move_stage action."""
    x: float
    y: float
    z: float

class StagePosition(TypedDict):
    """Stage position data."""
    x: float
    y: float
    z: float

class PictureData(TypedDict):
    """Picture data."""
    image_id: str


class SleepParams(BaseParams):
    """Parameters for sleep action."""
    seconds: int

class StagePositionResult(SuccessResult):
    """Result for get_stage_position action."""
    data: StagePosition

class PictureResult(SuccessResult):
    """Result for take_picture action."""
    data: PictureData

class MoveStageResult(SuccessResult):
    """Result for move_stage action."""
    data: Dict[str, float]

class TescanDevice(Device):
    """Tescan device implementation."""
    device_name = "tescan"

    @action("start_electron_beam")
    def start_electron_beam(self, params: BaseParams):
        from dp.agent.lab.tescan_element_operator import start_electron_beam as do_start_electron_beam
        do_start_electron_beam()
    
    @action("start_ion_beam")
    def start_ion_beam(self, params: BaseParams):
        from dp.agent.lab.tescan_element_operator import start_ion_beam as do_start_ion_beam
        do_start_ion_beam()
    
    @action("toggle_electron_scanning")
    def toggle_electron_scanning(self, params: BaseParams):
        """start scanning, this will cost 5 seconds to finish. call sleep(5) after toggle this.
        """
        from dp.agent.lab.tescan_element_operator import toggle_electron_scanning as do_toggle_electron_scanning
        do_toggle_electron_scanning()
    
    @action("toggle_ion_scanning")
    def toggle_ion_scanning(self, params: BaseParams):
        from dp.agent.lab.tescan_element_operator import toggle_ion_scanning as do_toggle_ion_scanning
        do_toggle_ion_scanning()

    @action("sleep")
    def sleep(self, params: SleepParams):
        time.sleep(params.get("seconds", 0))

    @action("take_picture")
    def take_picture(self, params: TakePictureParams) -> PictureResult:
        """Take a picture with the microscope.
        
        Args:
            params: Parameters for the action
                - horizontal_width: Horizontal width of the image
                
        Returns:
            Result of the action
        """
        time.sleep(1)
        
        hw = params.get("horizontal_width", "default")
        
        return PictureResult(
            message=f"Picture taken with {self.device_name} (width: {hw})",
            data={"image_id": "mock_image_123"}
        )
    
    @action("get_stage_position")
    def get_stage_position(self, params: GetStagePositionParams) -> StagePositionResult:
        """Get the stage position.
        
        Args:
            params: Parameters for the action (none required)
                
        Returns:
            Result of the action with x, y, z coordinates
        """
        time.sleep(0.5)
        
        return StagePositionResult(
            message=f"Stage position retrieved for {self.device_name}",
            data=StagePosition(
                x=10.5,
                y=20.3,
                z=5.1
            )
        )
    
    @action("move_stage")
    def move_stage(self, params: MoveStageParams) -> MoveStageResult:
        """Move the stage to a new position.
        
        Args:
            params: Parameters for the action
                - x: X coordinate
                - y: Y coordinate
                - z: Z coordinate
                
        Returns:
            Result of the action
        """
        time.sleep(2)
        
        if not all(k in params for k in ["x", "y", "z"]):
            # Use a proper MoveStageResult with error status instead of ErrorResult
            return MoveStageResult(
                message=f"Missing required parameters: x, y, z",
                data={}
            )
        
        try:
            new_position = {
                "x": float(params["x"]),
                "y": float(params["y"]),
                "z": float(params["z"])
            }
            
            return MoveStageResult(
                message=f"Stage moved for {self.device_name}",
                data=new_position
            )
        except (KeyError, TypeError, ValueError):
            # Use a proper MoveStageResult with error status instead of ErrorResult
            return MoveStageResult(
                message=f"Invalid parameters: x, y, z must be numeric values",
                data={}
            )
