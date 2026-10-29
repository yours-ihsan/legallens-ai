"""Central configuration for LegalLens AI.

All tunables are env-overridable so demo calibration does not require code edits.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DOCS_JSON = DATA_DIR / "docs.json"
USERS_JSON = DATA_DIR / "users.json"
SESSIONS_JSON = DATA_DIR / "sessions.json"
STATIC_DIR = BASE_DIR / "static"

for _dir in (UPLOADS_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

SESSION_COOKIE_NAME = "legallens_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days

def load_environment(override: bool = False) -> None:
    """Loads variables from local .env into os.environ if present."""
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        env_file = Path(".env").resolve()
    if not env_file.is_file():
        return

    try:
        import dotenv
        # If any target variable is empty in os.environ, populate it
        dotenv.load_dotenv(env_file, override=override)
    except Exception:
        pass

    # Direct fallback parsing in case python-dotenv is absent
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and v and (override or k not in os.environ or not os.environ[k].strip()):
                os.environ[k] = v
    except Exception:
        pass

# Ensure root .env is loaded before OAuth configuration is initialized
load_environment(override=False)

# Google OAuth 2.0 / OpenID Connect credentials (loaded from environment or .env)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()

COLLECTION_NAME = "legallens_chunks"

# LLM. If LEGALLENS_LLM_BASE_URL is set, any OpenAI-compatible HTTP API is used
# (works on a cloud server). If it is empty, the local OpenCode CLI is used
# (local development only).
MODEL = os.environ.get("LEGALLENS_MODEL", "opencode/mimo-v2.6-flash-free")
LLM_BASE_URL = os.environ.get("LEGALLENS_LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LEGALLENS_LLM_API_KEY", "")
LLM_TIMEOUT_SECONDS = int(os.environ.get("LEGALLENS_LLM_TIMEOUT", "180"))

# Retrieval / grounding
TOP_K = int(os.environ.get("LEGALLENS_TOP_K", "6"))
# Chat three-way decision: the LLM is consulted whenever the best retrieval
# score clears the LOW floor; weaker evidence is refused deterministically.
LOW_FLOOR = float(os.environ.get("LEGALLENS_LOW_FLOOR", "0.10"))
# Research still gates at the stricter threshold (best CASE score < this ->
# refusal, LLM not called).
REFUSAL_THRESHOLD = float(os.environ.get("LEGALLENS_THRESHOLD", "0.30"))
BAND_HIGH = float(os.environ.get("LEGALLENS_BAND_HIGH", "0.55"))
BAND_MEDIUM = float(os.environ.get("LEGALLENS_BAND_MEDIUM", "0.40"))

# Chunking (per-page so citations stay unambiguous)
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_QUESTION_CHARS = 2000