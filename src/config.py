import os
from dotenv import load_dotenv

load_dotenv()

# ===== LangSmith Tracing =====
# Set these in your .env file to enable tracing
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
    os.environ["LANGSMITH_TRACING"] = "true"
    print("[Config] LangSmith tracing ENABLED ✔")
else:
    print("[Config] LangSmith tracing DISABLED (no API key found)")

# Models (Using local Ollama models)
# future improvements ---> models like gpt 4o etc

WEAK_MODEL = "llama3.1:latest "
STRONG_MODEL = "gemma4:31b-cloud "
JUDGE_MODEL = "gemma4:31b-cloud "
CHALLENGER_MODEL = "gemma4:31b-cloud "

# RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHROMA_DB_DIR = "./chroma_db"
DATA_DIR = "./data"

# Graph Settings
MAX_RETRIES = 5  # Restored to 5 since local LLMs have no API cost
