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
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    mock_hardware: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
