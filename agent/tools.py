"""Tool definitions for Gemini function calling."""

from google.genai import types

# Tool function declarations for Gemini
TOOL_DECLARATIONS = [
    {
        "name": "pose_ctrl",
        "description": """Move the robot arm to a specific position in 3D space.
The coordinate system is based on the robot base:
- X: forward/backward distance from base (MUST be between 50 and 400mm)
- Y: left/right distance (MUST be between -400 and 400mm, positive = right)
- Z: height/up-down (MUST be between 0 and 300mm)
- T: gripper rotation angle (MUST be between 0 and 90 degrees)

IMPORTANT:
- Total reach sqrt(x² + y² + z²) MUST NOT exceed 500mm
- Values outside these ranges will be rejected
- Always check current position with pose_get first""",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "X coordinate in mm (forward/backward, valid range: 50-400)",
                },
                "y": {
                    "type": "number",
                    "description": "Y coordinate in mm (left/right, valid range: -400 to 400)",
                },
                "z": {
                    "type": "number",
                    "description": "Z coordinate in mm (height, valid range: 0-300)",
                },
                "t": {
                    "type": "number",
                    "description": "Gripper rotation angle in degrees (valid range: 0-90)",
                },
            },
            "required": ["x", "y", "z", "t"],
        },
    },
    {
        "name": "pose_get",
        "description": "Get the current position of the robot arm end effector. Returns the current x, y, z coordinates in mm and the gripper rotation angle t in degrees.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "move_home",
        "description": "Move the robot arm to its home/rest position. Use this to reset the arm position or when finished with a task.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "gripper_ctrl",
        "description": "Control the gripper to open or close it. Use angle 0 for fully closed (to grip objects) and 90 for fully open (to release objects).",
        "parameters": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Gripper angle in degrees (0 = closed, 90 = open)",
                }
            },
            "required": ["angle"],
        },
    },
]


def create_gemini_tools() -> list[types.Tool]:
    """Create Gemini Tool objects from declarations."""
    return [types.Tool(function_declarations=TOOL_DECLARATIONS)]
