import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    PORT = int(os.getenv("PORT", 5000))

    HOST = os.getenv("HOST", "0.0.0.0")

    # ============================================================
    # FILE UPLOAD SETTINGS
    # ============================================================

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

    MODELS_FOLDER = os.getenv("MODELS_FOLDER", "quantized_models")

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 20 * 1024 * 1024 * 1024))

    # Supported model formats
    ALLOWED_EXTENSIONS = {
        "onnx",
        "safetensors",
        "pt",
        "pth",
        "bin",
        "model",
        "zip",
        "json",
    }

    # ============================================================
    # MODEL SETTINGS
    # ============================================================

    DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "causal_lm")

    DEFAULT_QUANTIZATION = os.getenv("DEFAULT_QUANTIZATION", "int8")

    DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")

    # ============================================================
    # HUGGING FACE SETTINGS
    # ============================================================

    HF_TOKEN = os.getenv("HF_TOKEN", None)

    HF_ORGANIZATION = os.getenv("HF_ORGANIZATION", None)

    # ============================================================
    # LOGGING
    # ============================================================

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
