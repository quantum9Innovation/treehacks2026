#!/usr/bin/env python3
"""Entry point for the Gemini Robotics ER Robot Arm Control Agent (v3)."""

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
        description="Gemini Robotics ER Robot Arm Control Agent (v3)"
    )

    # API key
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API key (or set GOOGLE_API_KEY env var)",
    )

    # Model
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=1024,
        help="Thinking budget for Gemini ER reasoning (default: 1024)",
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

    # Validate API key (not needed for hardware-only modes)
    if not args.calibrate and not args.touch_debug:
        if not args.google_api_key:
            print("Error: Google API key required.")
            print("Set GOOGLE_API_KEY environment variable or use --google-api-key")
            return 1

    # Resolve calibration path (shared with agent_v2)
    calibration_path = (
        Path(args.calibration)
        if args.calibration
        else (Path(__file__).parent.parent / "agent_v2" / "calibration_data.json")
    )

    try:
        if args.calibrate:
            # Calibration-only mode (reuses v2 calibration code)
            import time
            from agent.camera import RealSenseCamera
            from motion_controller.motion import Motion
            from agent_v2.calibration import CalibrationProcedure
            from agent_v2.coordinate_transform import CoordinateTransform

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
            # Touch debug mode — no LLM needed
            from .agent import AgentV3

            agent = AgentV3(
                google_api_key="unused",
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
        from .agent import AgentV3

        agent = AgentV3(
            google_api_key=args.google_api_key,
            arm_port=args.port,
            auto_confirm=args.auto_confirm,
            debug=args.debug,
            thinking_budget=args.thinking_budget,
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
