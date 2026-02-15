"""Tool declarations for Gemini function calling (Agent V3)."""

from google.genai import types

TOOL_DECLARATIONS = [
    {
        "name": "look",
        "description": (
            "Capture a fresh camera frame. Returns the raw color image (640x480) "
            "and a colorized depth image so you can see the scene. After calling "
            "look(), you can call detect(query) to locate objects."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "detect",
        "description": (
            "Detect and segment an object using a natural language description. "
            "Returns the object's pixel centroid, bounding box, segmentation mask "
            "overlay, and 3D arm coordinates. Use this to precisely locate an "
            "object before moving to it. Requires a prior look() call. "
            "Example queries: 'the red cup', 'the nearest object', 'the pen on the left'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of the object to detect. "
                        "Be specific: 'the red cup' is better than 'cup'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "goto_pixel",
        "description": (
            "Move the robot arm to the 3D position corresponding to a pixel "
            "coordinate. The pixel is deprojected using the depth camera and "
            "transformed to arm coordinates via calibration. An optional Z offset "
            "(default 50mm) is added to hover above the target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pixel_x": {
                    "type": "integer",
                    "description": "X pixel coordinate to move to",
                },
                "pixel_y": {
                    "type": "integer",
                    "description": "Y pixel coordinate to move to",
                },
                "z_offset_mm": {
                    "type": "number",
                    "description": "Height offset above the target in mm (default: 50). Use 0 to go directly to target height.",
                },
            },
            "required": ["pixel_x", "pixel_y"],
        },
    },
    {
        "name": "pose_get",
        "description": (
            "Get the current position of the robot arm end effector. "
            "Returns the current x, y, z coordinates in mm. "
            "Z is ground-relative (0 = ground level, positive = up)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "move_home",
        "description": (
            "Move the robot arm to its home/rest position. "
            "Use this to reset the arm position or when finished with a task."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "gripper_ctrl",
        "description": (
            "Control the gripper to open or close it. "
            "Use angle 0 for fully closed (to grip objects) and 90 for fully open."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Gripper angle in degrees (0 = closed, 90 = open)",
                },
            },
            "required": ["angle"],
        },
    },
]


def create_gemini_tools() -> list[types.Tool]:
    """Create Gemini Tool objects from declarations."""
    return [types.Tool(function_declarations=TOOL_DECLARATIONS)]
