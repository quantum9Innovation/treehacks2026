#!/usr/bin/env python3
"""Entry point for the VLM Robot Arm Control Agent."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .vlm_agent import VLMRobotAgent


def main() -> int:
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    parser = argparse.ArgumentParser(
        description="Run the VLM Robot Arm Control Agent (via Helicone)"
    )
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
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model ID (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for robot arm (auto-detect if omitted)",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=300.0,
        help="Z offset for inverted mounting in mm (default: 300)",
    )
    parser.add_argument(
        "--no-invert-z",
        action="store_true",
        help="Disable Z-axis inversion (if robot is not mounted upside down)",
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip user confirmation for movements (DANGEROUS - use with caution!)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Thinking effort level for reasoning models (default: low)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for troubleshooting",
    )
    args = parser.parse_args()

    if not args.openai_api_key:
        print("Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --openai-api-key")
        return 1

    if not args.helicone_api_key:
        print("Error: Helicone API key required.")
        print("Set HELICONE_API_KEY environment variable or use --helicone-api-key")
        return 1

    try:
        agent = VLMRobotAgent(
            openai_api_key=args.openai_api_key,
            helicone_api_key=args.helicone_api_key,
            model=args.model,
            arm_port=args.port,
            z_offset=args.z_offset,
            invert_z=not args.no_invert_z,
            auto_confirm=args.auto_confirm,
            debug=args.debug,
            reasoning_effort=args.reasoning_effort,
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
