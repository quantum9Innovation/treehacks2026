"""LLM tool declarations for the object-aware agent."""

TOOL_DECLARATIONS = [
    {
        "name": "describe_scene",
        "description": (
            "Capture a fresh camera frame, run object detection, and return "
            "an annotated image with bounding boxes plus a structured JSON list "
            "of all detected objects with their IDs, labels, and 3D arm coordinates. "
            "Call this first to see what objects are in the scene before deciding "
            "what to do."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "goto",
        "description": (
            "Move the robot arm to hover above a detected object. "
            "The object must have been detected in the most recent describe_scene() call. "
            "The arm moves to the object's 3D position with an optional Z offset "
            "(default 50mm above) to avoid collision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "integer",
                    "description": "ID of the detected object to move to (from describe_scene output)",
                },
                "z_offset_mm": {
                    "type": "number",
                    "description": "Height offset above the object in mm (default: 50). Use 0 to go directly to object height.",
                },
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "pose_get",
        "description": (
            "Get the current position of the robot arm end effector. "
            "Returns the current x, y, z coordinates in mm and the gripper rotation angle t in degrees."
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
