import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Model Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "qwen/qwen3.8-27b"

# Fast, lightweight local CPU embedding model (runs completely free offline)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"