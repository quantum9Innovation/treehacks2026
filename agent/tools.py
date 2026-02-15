"""Tool declarations for Agent V3 (provider-agnostic)."""

TOOL_DECLARATIONS = [
    {
        "name": "detect",
        "description": (
            "Detect and segment an object using a natural language description. "
            "Returns the object's pixel centroid, bounding box, segmentation mask "
            "overlay, and 3D arm coordinates. Use this to discover and roughly locate "
            "objects by description. Uses the latest auto-captured camera frame. "
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
        "name": "segment",
        "description": (
            "Run SAM2 point-prompt segmentation at a pixel coordinate. "
            "Returns a mask overlay image and the 3D arm coordinates at the "
            "exact pixel you specified. More precise than detect() — use this "
            "to visually confirm you are targeting the right object before moving. "
            "Uses the latest auto-captured camera frame."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pixel_x": {
                    "type": "integer",
                    "description": "X pixel coordinate to segment at (0 = left, 639 = right)",
                },
                "pixel_y": {
                    "type": "integer",
                    "description": "Y pixel coordinate to segment at (0 = top, 479 = bottom)",
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
        "name": "move_to_xyz",
        "description": (
            "Move the robot arm to an absolute position in arm coordinates (mm). "
            "Z is ground-relative (0 = ground, positive = up). Use coordinates "
            "from detect(), segment(), or pose_get(). Faster than goto_pixel "
            "because it skips depth lookup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "X coordinate in mm (arm frame)",
                },
                "y": {
                    "type": "number",
                    "description": "Y coordinate in mm (arm frame)",
                },
                "z": {
                    "type": "number",
                    "description": "Z coordinate in mm, ground-relative (0 = ground, positive = up)",
                },
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "move_relative",
        "description": (
            "Move the robot arm by a relative offset from its current position. "
            "dx, dy, dz are in mm. Positive dz moves up, negative moves down. "
            "Useful for fine adjustments, lateral sweeps, and step-by-step movements."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dx": {"type": "number", "description": "Relative X offset in mm"},
                "dy": {"type": "number", "description": "Relative Y offset in mm"},
                "dz": {
                    "type": "number",
                    "description": "Relative Z offset in mm (positive = up)",
                },
            },
            "required": ["dx", "dy", "dz"],
        },
    },
    {
        "name": "press_down",
        "description": (
            "Lower the arm downward with force monitoring. Descends until contact "
            "is detected (torque exceeds threshold) or max distance reached. "
            "Returns whether contact was made and final position. "
            "Use for pressing, cutting, poking, and controlled descent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "max_force": {
                    "type": "integer",
                    "description": "Force threshold (torque delta) to stop. Default 50 (light contact). Use 30 for delicate, 80 for firm.",
                },
                "max_distance_mm": {
                    "type": "number",
                    "description": "Maximum descent distance in mm (default 50).",
                },
            },
        },
    },
    {
        "name": "force_move",
        "description": (
            "Move the arm in a specified direction with force/torque monitoring. "
            "Moves along the direction vector until contact is detected or max "
            "distance reached. Use for pushing objects, lateral probing, and "
            "directional force application. Direction is normalized internally."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dx": {
                    "type": "number",
                    "description": "Direction X component (will be normalized)",
                },
                "dy": {
                    "type": "number",
                    "description": "Direction Y component (will be normalized)",
                },
                "dz": {
                    "type": "number",
                    "description": "Direction Z component (will be normalized)",
                },
                "max_force": {
                    "type": "integer",
                    "description": "Torque delta threshold (default 50)",
                },
                "max_distance_mm": {
                    "type": "number",
                    "description": "Maximum travel distance in mm (default 100)",
                },
            },
            "required": ["dx", "dy", "dz"],
        },
    },
    {
        "name": "execute_trajectory",
        "description": (
            "Execute a smooth path through a sequence of XYZ waypoints at ~50Hz. "
            "Use for cutting motions, wiping, drawing, or any continuous path. "
            "Each waypoint is [x, y, z] in ground-relative arm coordinates (mm). "
            "Space waypoints 5-10mm apart for smooth motion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "waypoints": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "description": "List of [x, y, z] waypoints in mm. Minimum 2 waypoints.",
                },
                "force_threshold": {
                    "type": "integer",
                    "description": "If set, monitor torque and stop on contact. Use 80-100 (higher than press_down due to motion noise).",
                },
            },
            "required": ["waypoints"],
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


def get_tool_declarations(
    has_sam2: bool = True, has_gemini_vision: bool = True
) -> list[dict]:
    """Get tool declarations filtered by available vision backends.

    Args:
        has_sam2: Whether SAM2 is available (enables segment tool).
        has_gemini_vision: Whether Gemini ER vision is available (enables detect tool).

    Returns:
        Filtered list of tool declaration dicts.
    """
    exclude = set()
    if not has_sam2:
        exclude.add("segment")
    if not has_gemini_vision:
        exclude.add("detect")
    return [t for t in TOOL_DECLARATIONS if t["name"] not in exclude]
