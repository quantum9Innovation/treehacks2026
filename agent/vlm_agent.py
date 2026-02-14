"""OpenAI/OpenRouter VLM-based robot arm control agent."""

import base64
import json
import logging
import time
from typing import Any

from openai import OpenAI

from .arm_controller import RobotArmController
from .camera import RealSenseCamera
from .prompts import SYSTEM_PROMPT, create_task_prompt
from .tools import TOOL_DECLARATIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_agent")

# Helicone proxy configuration
HELICONE_BASE_URL = "https://oai.helicone.ai/v1"


def create_openai_client(
    openai_api_key: str,
    helicone_api_key: str,
    debug: bool = False,
) -> OpenAI:
    """
    Create an OpenAI client configured for Helicone proxy.

    Args:
        openai_api_key: OpenAI API key
        helicone_api_key: Helicone API key for tracking
        debug: Enable debug logging

    Returns:
        Configured OpenAI client
    """
    if debug:
        logging.getLogger("vlm_agent").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)

    logger.info(f"Using Helicone proxy at {HELICONE_BASE_URL}")
    return OpenAI(
        api_key=openai_api_key,
        base_url=HELICONE_BASE_URL,
        default_headers={
            "Helicone-Auth": f"Bearer {helicone_api_key}",
        },
    )


def convert_tools_to_openai_format() -> list[dict]:
    """Convert tool declarations to OpenAI function calling format."""
    tools = []
    for decl in TOOL_DECLARATIONS:
        tools.append({
            "type": "function",
            "function": {
                "name": decl["name"],
                "description": decl["description"],
                "parameters": decl["parameters"],
            },
        })
    return tools


class ConfirmationHandler:
    """Handles user confirmation before executing robot arm movements."""

    def __init__(self, auto_confirm: bool = False):
        self.auto_confirm = auto_confirm

    def format_action(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format the pending action for display."""
        if tool_name == "pose_ctrl":
            return (
                f"MOVE ARM to position:\n"
                f"  X: {args['x']:.1f} mm (forward/back)\n"
                f"  Y: {args['y']:.1f} mm (left/right)\n"
                f"  Z: {args['z']:.1f} mm (up/down)\n"
                f"  T: {args['t']:.1f} deg (gripper rotation)"
            )
        elif tool_name == "move_home":
            return "MOVE ARM to HOME position"
        elif tool_name == "gripper_ctrl":
            state = "OPEN" if args["angle"] > 45 else "CLOSE"
            return f"GRIPPER {state} to {args['angle']:.1f} degrees"
        elif tool_name == "pose_get":
            return "READ current arm position (no movement)"
        else:
            return f"Execute {tool_name} with args: {json.dumps(args)}"

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if this tool requires user confirmation."""
        return tool_name in ["pose_ctrl", "move_home", "gripper_ctrl"]

    def request_confirmation(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Request user confirmation for a pending action."""
        if self.auto_confirm:
            return True

        if not self.requires_confirmation(tool_name):
            return True

        action_str = self.format_action(tool_name, args)
        print("\n" + "=" * 50)
        print("PENDING ACTION:")
        print(action_str)
        print("=" * 50)

        while True:
            response = input("Execute this action? [y/n/q]: ").strip().lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            elif response in ["q", "quit"]:
                raise KeyboardInterrupt("User requested quit")
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")


class VLMRobotAgent:
    """Helicone VLM-based robot arm control agent."""

    def __init__(
        self,
        openai_api_key: str,
        helicone_api_key: str,
        model: str = "gpt-4.1-mini",
        arm_port: str | None = None,
        z_offset: float = 300.0,
        invert_z: bool = True,
        auto_confirm: bool = False,
        debug: bool = False,
        reasoning_effort: str = "low",
    ):
        """
        Initialize the VLM robot agent.

        Args:
            openai_api_key: OpenAI API key
            helicone_api_key: Helicone API key for tracking
            model: Model to use
            arm_port: Serial port for robot arm (auto-detect if None)
            z_offset: Z offset for inverted mounting (mm)
            invert_z: Whether Z axis is inverted
            auto_confirm: Skip user confirmation (dangerous!)
            debug: Enable debug logging
            reasoning_effort: Thinking effort level (low, medium, high)
        """
        self.debug = debug
        self.reasoning_effort = reasoning_effort
        logger.info(f"Initializing VLMRobotAgent with model={model}")

        # Initialize OpenAI client via Helicone proxy
        self.client = create_openai_client(openai_api_key, helicone_api_key, debug=debug)
        self.model = model

        # Initialize hardware
        self.camera = RealSenseCamera()
        self.arm = RobotArmController(
            port=arm_port,
            z_offset=z_offset,
            invert_z=invert_z,
        )

        # Initialize confirmation handler
        self.confirmation = ConfirmationHandler(auto_confirm=auto_confirm)

        # Create tools configuration (OpenAI format)
        self.tools = convert_tools_to_openai_format()

        # Tool execution map
        self.tool_executors = {
            "pose_ctrl": self._execute_pose_ctrl,
            "pose_get": self._execute_pose_get,
            "move_home": self._execute_move_home,
            "gripper_ctrl": self._execute_gripper_ctrl,
        }

        # Note: We don't persist history across tasks to avoid orphaned tool messages
        pass

    def clear_history(self) -> None:
        """Clear conversation history (no-op, history not persisted)."""
        logger.info("History cleared (no-op)")

    def start(self) -> None:
        """Start the agent (initialize hardware)."""
        print("Starting camera...")
        self.camera.start()
        print("Moving arm to home position...")
        self.arm.move_home()
        time.sleep(1)
        print("Agent ready.")

    def stop(self) -> None:
        """Stop the agent (cleanup hardware)."""
        print("\nStopping agent...")
        self.arm.move_home()
        time.sleep(1)
        self.camera.stop()
        print("Agent stopped.")

    def _execute_pose_ctrl(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute pose_ctrl tool."""
        return self.arm.pose_ctrl(
            x=args["x"],
            y=args["y"],
            z=args["z"],
            t=args["t"],
        )

    def _execute_pose_get(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute pose_get tool."""
        return self.arm.pose_get()

    def _execute_move_home(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute move_home tool."""
        return self.arm.move_home()

    def _execute_gripper_ctrl(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute gripper_ctrl tool."""
        return self.arm.gripper_ctrl(angle=args["angle"])

    def _execute_tool(self, tool_call) -> dict[str, Any]:
        """Execute a tool call with user confirmation."""
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

        # Request confirmation for movement commands
        if not self.confirmation.request_confirmation(tool_name, args):
            return {"status": "cancelled", "message": "User cancelled the action"}

        # Execute the tool
        executor = self.tool_executors.get(tool_name)
        if executor is None:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        try:
            print(f"Executing {tool_name}...")
            result = executor(args)
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _strip_old_images(self, messages: list[dict]) -> None:
        """Remove image content from previous user messages to reduce payload size."""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # Replace image_url items with placeholder text
            new_content = []
            had_images = False
            for item in content:
                if item.get("type") == "image_url":
                    had_images = True
                else:
                    new_content.append(item)
            if had_images:
                new_content.insert(0, {"type": "text", "text": "[Previous images removed]"})
                msg["content"] = new_content

    def _create_image_content(self) -> list[dict]:
        """Capture camera frames and create OpenAI vision content."""
        color_bytes, depth_bytes = self.camera.capture_as_bytes()

        # Encode images as base64
        color_b64 = base64.b64encode(color_bytes).decode("utf-8")
        depth_b64 = base64.b64encode(depth_bytes).decode("utf-8")

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{color_b64}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": "[Color camera image above, depth image below]",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{depth_b64}",
                    "detail": "high",
                },
            },
        ]

    def process_task(self, task: str) -> str:
        """
        Process a task using the VLM agent loop.

        Args:
            task: Natural language task description

        Returns:
            Final response from the agent
        """
        logger.info(f"Processing task: {task}")

        # Capture current camera frame
        print("Capturing camera frame...")
        image_content = self._create_image_content()
        logger.debug(f"Captured {len(image_content)} image parts")

        # Create initial prompt
        task_prompt = create_task_prompt(task)
        logger.debug(f"Task prompt length: {len(task_prompt)} chars")

        # Build initial user message with images and task
        user_message = {
            "role": "user",
            "content": image_content + [{"type": "text", "text": task_prompt}],
        }

        # Build messages list (fresh for each task)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ]

        # Agent loop
        max_iterations = 10
        iteration = 0
        empty_response_retries = 0
        max_empty_retries = 1

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Agent iteration {iteration} ---")
            logger.debug(f"Sending request with {len(messages)} messages")

            # Get model response
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools,
                    "tool_choice": "auto",
                }

                # Add reasoning effort for models that support it
                if "codex" in self.model.lower() or "o1" in self.model.lower() or "o3" in self.model.lower():
                    request_kwargs["extra_body"] = {
                        "reasoning_effort": self.reasoning_effort,
                    }

                response = self.client.chat.completions.create(**request_kwargs)
                logger.debug(f"Response received: {type(response)}")
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

            # Check for empty response
            if not response.choices:
                logger.warning("No choices in response")

                if empty_response_retries < max_empty_retries:
                    empty_response_retries += 1
                    logger.info(f"Retrying (attempt {empty_response_retries})")
                    print("No response from model, retrying...")
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        user_message,
                    ]
                    continue

                return "Task could not be completed (model returned empty response)."

            choice = response.choices[0]
            assistant_message = choice.message
            logger.debug(f"Finish reason: {choice.finish_reason}")

            # Print any text content
            if assistant_message.content:
                print(f"\nAgent: {assistant_message.content}")

            # Add assistant message to conversation
            messages.append(assistant_message.model_dump())

            # Check if there are tool calls
            if not assistant_message.tool_calls:
                # No tool calls - we have a final response
                return assistant_message.content or "Task completed."

            # Execute tool calls and collect results
            movement_executed = False

            for tool_call in assistant_message.tool_calls:
                result = self._execute_tool(tool_call)
                print(f"Result: {result}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

                tool_name = tool_call.function.name
                if tool_name in ["pose_ctrl", "move_home", "gripper_ctrl"]:
                    if result.get("status") == "success":
                        movement_executed = True

            # Capture new frame after movement
            if movement_executed:
                print("Capturing updated camera view...")
                time.sleep(0.5)

                # # Strip old images from previous messages to reduce payload
                # self._strip_old_images(messages)

                new_image_content = self._create_image_content()
                messages.append({
                    "role": "user",
                    "content": new_image_content + [
                        {"type": "text", "text": "[Updated camera view after movement. Confirm task completion or continue if more actions needed.]"}
                    ],
                })
                time.sleep(0.3)

        return "Maximum iterations reached. Task may be incomplete."

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        self.start()

        print("\n" + "=" * 60)
        print("VLM Robot Agent - Interactive Mode")
        print("=" * 60)
        print("Enter tasks for the robot, or commands:")
        print("  quit    - Exit the agent")
        print("  home    - Move arm to home position")
        print("  pos     - Show current arm position")
        print("  clear   - Clear conversation history")
        print()
        print("Example tasks:")
        print('  "Describe what you see"')
        print('  "Move the arm forward 50mm"')
        print('  "Pick up the red object"')
        print()

        try:
            while True:
                task = input("Task> ").strip()

                if not task:
                    continue

                if task.lower() in ["quit", "exit", "q"]:
                    break

                if task.lower() == "home":
                    if self.confirmation.request_confirmation("move_home", {}):
                        self.arm.move_home()
                        print("Moved to home position.")
                    continue

                if task.lower() == "pos":
                    pos = self.arm.pose_get()
                    print(f"Current position: {pos}")
                    continue

                if task.lower() == "clear":
                    self.clear_history()
                    print("Conversation history cleared.")
                    continue

                print("\nProcessing task...")
                try:
                    response = self.process_task(task)
                    print(f"\n{'=' * 60}")
                    print("Task completed.")
                    print(f"{'=' * 60}\n")
                except Exception as e:
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    print(f"\nError processing task: {e}\n")

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.stop()
