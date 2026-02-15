#!/usr/bin/env python3
"""Entry point for Agent V3 — multi-provider robot arm control agent."""

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
        description="Agent V3 — Multi-provider robot arm control agent"
    )

    # LLM provider
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        default="openai",
        help="LLM provider for reasoning (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (default: gpt-5.2 for openai, gemini-robotics-er-1.5-preview for gemini)",
    )

    # OpenAI-specific
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
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort for OpenAI reasoning models (default: low)",
    )

    # Gemini-specific
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=1024,
        help="Thinking budget for Gemini reasoning (default: 1024)",
    )

    # Vision backends
    parser.add_argument(
        "--no-sam2",
        action="store_true",
        help="Disable SAM2 vision (segment tool unavailable)",
    )
    parser.add_argument(
        "--sam2-model",
        type=str,
        choices=["tiny", "small", "base_plus", "large"],
        default="tiny",
        help="SAM2 model size (default: tiny)",
    )
    parser.add_argument(
        "--no-gemini-vision",
        action="store_true",
        help="Disable Gemini ER vision (detect tool unavailable)",
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

    # Resolve calibration path (shared with agent_v2)
    calibration_path = (
        Path(args.calibration)
        if args.calibration
        else (Path(__file__).parent.parent / "agent_v2" / "calibration_data.json")
    )

    try:
        # ── Calibration-only mode ──
        if args.calibrate:
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

        # ── Touch debug mode (no LLM needed) ──
        if args.touch_debug:
            from .agent import AgentV3
            from .llm import GeminiProvider  # dummy, unused for touch debug

            # Need a provider for the constructor — create a minimal one
            # Touch debug doesn't call the LLM so the key doesn't matter
            provider = GeminiProvider(
                google_api_key="unused",
                model="gemini-robotics-er-1.5-preview",
            )

            agent = AgentV3(
                provider=provider,
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

        # ── Normal agent mode — build LLM provider ──
        if args.provider == "openai":
            if not args.openai_api_key:
                print("Error: OpenAI API key required for --provider openai.")
                print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
                return 1
            if not args.helicone_api_key:
                print("Error: Helicone API key required for --provider openai.")
                print("Set HELICONE_API_KEY environment variable or use --helicone-api-key")
                return 1

            from .llm import OpenAIProvider

            model = args.model or "gpt-5.2"
            print(f"Initializing OpenAI provider (model={model})...")
            provider = OpenAIProvider(
                openai_api_key=args.openai_api_key,
                helicone_api_key=args.helicone_api_key,
                model=model,
                debug=args.debug,
                reasoning_effort=args.reasoning_effort,
            )

        elif args.provider == "gemini":
            if not args.google_api_key:
                print("Error: Google API key required for --provider gemini.")
                print("Set GOOGLE_API_KEY environment variable or use --google-api-key")
                return 1

            from .llm import GeminiProvider

            model = args.model or "gemini-robotics-er-1.5-preview"
            print(f"Initializing Gemini provider (model={model})...")
            provider = GeminiProvider(
                google_api_key=args.google_api_key,
                model=model,
                thinking_budget=args.thinking_budget,
            )

        # ── Build vision backends ──
        sam2_backend = None
        gemini_vision = None

        if not args.no_sam2:
            from agent_v2.vision.sam2_backend import SAM2Backend

            print(f"Loading SAM2 model ({args.sam2_model})...")
            sam2_backend = SAM2Backend(model_size=args.sam2_model)

        if not args.no_gemini_vision:
            if args.google_api_key:
                from google import genai
                from .gemini_vision import GeminiVision

                print("Initializing Gemini ER vision...")
                vision_client = genai.Client(api_key=args.google_api_key)
                gemini_vision = GeminiVision(client=vision_client)
            else:
                print("Note: Gemini vision disabled (no Google API key).")

        # ── Create and run agent ──
        from .agent import AgentV3

        agent = AgentV3(
            provider=provider,
            arm_port=args.port,
            auto_confirm=args.auto_confirm,
            debug=args.debug,
            calibration_path=calibration_path,
            sam2_backend=sam2_backend,
            gemini_vision=gemini_vision,
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
