"""LLM tool declarations for the vision-driven SAM2 agent."""

TOOL_DECLARATIONS = [
    {
        "name": "look",
        "description": (
            "Capture a fresh camera frame and encode it for segmentation. "
            "Returns the raw color image and a colorized depth image so you can "
            "see the scene. No objects are detected — you decide what to interact "
            "with by examining the images. After calling look(), you can call "
            "segment(pixel_x, pixel_y) on any point of interest."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "segment",
        "description": (
            "Run SAM2 point-prompt segmentation at a pixel coordinate. "
            "Returns a mask overlay image, the mask's centroid pixel, and the "
            "corresponding 3D arm coordinates. Use this to precisely locate an "
            "object before moving to it. Requires a prior look() call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pixel_x": {
                    "type": "integer",
                    "description": "X pixel coordinate to segment at (0 = left edge, 639 = right edge)",
                },
                "pixel_y": {
                    "type": "integer",
                    "description": "Y pixel coordinate to segment at (0 = top edge, 479 = bottom edge)",
                },
            },
            "required": ["pixel_x", "pixel_y"],
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
