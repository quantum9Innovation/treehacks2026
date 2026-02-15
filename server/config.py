"""Server configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    helicone_api_key: str = ""
    arm_devices: list[str] = ["/dev/ARM0", "/dev/ARM1", "/dev/ARM2", "/dev/ARM3"]
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
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
