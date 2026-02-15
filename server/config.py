"""Server configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    helicone_api_key: str = ""
    arm_port: str | None = None
    sam2_model: str = "tiny"
    sam2_device: str = "auto"
    calibration_path: str = "agent_v2/calibration_data.json"
    llm_model: str = "gpt-5.2"
    reasoning_effort: str = "low"
    host: str = "0.0.0.0"
    port: int = 8420
    cors_origins: list[str] = ["*"]
    mock_hardware: bool = False
    google_api_key: str = ""
    llm_provider: str = "openai"  # "openai" or "gemini"
    gemini_model: str = "gemini-robotics-er-1.5-preview"
    gemini_thinking_budget: int = 1024
    enable_sam2: bool = True
    enable_gemini_vision: bool = True
    webcam_device: int = 0  # /dev/video index for j5 webcam
    webcam_width: int = 1920
    webcam_height: int = 1080
    webcam_fps: int = 30
    webcam_jpeg_quality: int = 85

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
