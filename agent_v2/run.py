#!/usr/bin/env python3
"""Entry point for the Object-Aware VLM Robot Arm Control Agent (v2)."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    parser = argparse.ArgumentParser(
        description="Object-Aware VLM Robot Arm Control Agent (v2)"
    )

    # Vision backend
    parser.add_argument(
        "--vision",
        type=str,
        choices=["yolo", "mock"],
        default="yolo",
        help="Vision backend to use (default: yolo)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model file (default: yolo11n.pt, also: yolo11n-seg.pt)",
    )

    # Calibration
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration JSON (default: agent_v2/calibration_data.json)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run interactive calibration procedure and exit",
    )
    parser.add_argument(
        "--touch-debug",
        action="store_true",
        help="Run touch debug mode (click to move arm, then reset)",
    )

    # API keys
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--helicone-api-key",
        type=str,
        default=os.environ.get("HELICONE_API_KEY"),
        help="Helicone API key (or set HELICONE_API_KEY env var)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model ID (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Thinking effort level for reasoning models (default: low)",
    )

    # Hardware
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for robot arm (auto-detect if omitted)",
    )

    # Behavior
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip user confirmation for movements (DANGEROUS)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate API keys (not needed for hardware-only modes)
    if not args.calibrate and not args.touch_debug:
        if not args.openai_api_key:
            print("Error: OpenAI API key required.")
            print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
            return 1
        if not args.helicone_api_key:
            print("Error: Helicone API key required.")
            print("Set HELICONE_API_KEY environment variable or use --helicone-api-key")
            return 1

    # Resolve calibration path
    calibration_path = Path(args.calibration) if args.calibration else (
        Path(__file__).parent / "calibration_data.json"
    )

    # Create vision backend
    if args.vision == "yolo":
        from .vision.yolo_backend import YOLOBackend
        vision_backend = YOLOBackend(model_path=args.yolo_model)
    else:
        from .vision.mock_backend import MockBackend
        vision_backend = MockBackend()

    try:
        if args.calibrate:
            # Calibration-only mode
            import time
            from agent.camera import RealSenseCamera
            from motion_controller.motion import Motion
            from .calibration import CalibrationProcedure
            from .coordinate_transform import CoordinateTransform

            camera = RealSenseCamera()
            camera.start()
            motion = Motion(port=args.port, inverted=True)
            ct = CoordinateTransform(calibration_path=calibration_path)
            ct.set_intrinsics_from_camera(camera)

            print("Probing ground level...")
            motion.probe_ground()

            try:
                proc = CalibrationProcedure(camera, motion, ct)
                proc.run(save_path=calibration_path)
            finally:
                motion.home()
                time.sleep(1)
                camera.stop()
            return 0

        if args.touch_debug:
            # Touch debug mode — no LLM needed, pass dummy keys
            from .agent import AgentV2

            agent = AgentV2(
                openai_api_key="unused",
                helicone_api_key="unused",
                vision_backend=vision_backend,
                model=args.model,
                arm_port=args.port,
                auto_confirm=True,
                debug=args.debug,
                calibration_path=calibration_path,
            )
            agent.start()
            try:
                agent._run_touch_debug()
            finally:
                agent.stop()
            return 0

        # Normal agent mode
        from .agent import AgentV2

        agent = AgentV2(
            openai_api_key=args.openai_api_key,
            helicone_api_key=args.helicone_api_key,
            vision_backend=vision_backend,
            model=args.model,
            arm_port=args.port,
            auto_confirm=args.auto_confirm,
            debug=args.debug,
            reasoning_effort=args.reasoning_effort,
            calibration_path=calibration_path,
        )
        agent.run_interactive()
        return 0

    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
