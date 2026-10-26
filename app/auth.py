"""Authentication and session management for LegalLens AI.

Provides:
- PBKDF2-HMAC-SHA256 password hashing with 100,000 iterations and 16-byte random salt.
- Constant-time password comparison to prevent timing attacks.
- Thread-safe JSON-backed user and session persistence.
- Cryptographically secure session tokens.
- Input validation for name, email, and password.
- Google OAuth 2.0 / OpenID Connect user account linking and creation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import secrets
import threading
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import SESSIONS_JSON, USERS_JSON, SESSION_MAX_AGE_SECONDS

_auth_lock = threading.Lock()
_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_MIN_PASSWORD_LENGTH = 8
_MAX_PASSWORD_LENGTH = 128
_PBKDF2_ITERATIONS = 100_000


class AuthUser(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    google_id: Optional[str] = None
    created_at: Optional[str] = None


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=128)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class RegisterResponse(BaseModel):
    status: str = "registered"
    message: str
    user: AuthUser


class AuthResponse(BaseModel):
    user: AuthUser
    token: str


def _load_json_file(path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_json_file(path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """Hashes password with PBKDF2-HMAC-SHA256 and 100,000 rounds.

    Returns (salt_hex, hash_hex).
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return salt.hex(), pw_hash.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Verifies a password against stored salt and hash using constant-time comparison."""
    try:
        salt = bytes.fromhex(salt_hex)
        _, check_hash = hash_password(password, salt=salt)
        return secrets.compare_digest(check_hash, hash_hex)
    except Exception:
        return False


def validate_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("Name cannot be empty.")
    if len(cleaned) > 100:
        raise ValueError("Name is too long (maximum 100 characters).")
    return cleaned


def validate_email(email: str) -> str:
    cleaned = (email or "").strip()
    if not cleaned:
        raise ValueError("Email address cannot be empty.")
    if len(cleaned) > 255:
        raise ValueError("Email address is too long.")
    if not _EMAIL_REGEX.match(cleaned):
        raise ValueError("Please enter a valid email address.")
    return cleaned.lower()


def validate_password(password: str) -> str:
    if len(password) < _MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {_MIN_PASSWORD_LENGTH} characters long.")
    if len(password) > _MAX_PASSWORD_LENGTH:
        raise ValueError("Password is too long.")
    return password


def register_user(name: str, email: str, password: str) -> AuthUser:
    """Registers a new user and persists to users.json."""
    clean_name = validate_name(name)
    clean_email = validate_email(email)
    validate_password(password)

    with _auth_lock:
        users = _load_json_file(USERS_JSON)
        if clean_email in users:
            raise ValueError("An account with this email address already exists.")

        salt_hex, hash_hex = hash_password(password)
        user_id = uuid.uuid4().hex[:12]
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        users[clean_email] = {
            "user_id": user_id,
            "name": clean_name,
            "email": clean_email,
            "salt": salt_hex,
            "password_hash": hash_hex,
            "google_id": None,
            "created_at": now_iso,
        }
        _save_json_file(USERS_JSON, users)

    return AuthUser(user_id=user_id, name=clean_name, email=clean_email, created_at=now_iso)


def authenticate_user(email: str, password: str) -> Optional[AuthUser]:
    """Authenticates a user by email and password."""
    clean_email = (email or "").strip().lower()
    with _auth_lock:
        users = _load_json_file(USERS_JSON)
        user_record = users.get(clean_email)

    if not user_record:
        # Run dummy verification to avoid timing attacks
        _ = hash_password(password, salt=b"dummy_salt_12345")
        return None

    salt_hex = user_record.get("salt")
    hash_hex = user_record.get("password_hash")
    if not salt_hex or not hash_hex:
        # Account was created with Google without a password
        return None

    if not verify_password(password, salt_hex, hash_hex):
        return None

    return AuthUser(
        user_id=user_record["user_id"],
        name=user_record.get("name"),
        email=user_record["email"],
        google_id=user_record.get("google_id"),
        created_at=user_record.get("created_at"),
    )


def get_user_by_email(email: str) -> Optional[AuthUser]:
    """Retrieves an existing registered user by email (case-insensitive)."""
    clean_email = (email or "").strip().lower()
    if not clean_email:
        return None
    with _auth_lock:
        users = _load_json_file(USERS_JSON)
        user_record = users.get(clean_email)
        if not user_record:
            for k, v in users.items():
                if k.lower() == clean_email or (isinstance(v, dict) and v.get("email", "").lower() == clean_email):
                    user_record = v
                    break
        if not user_record:
            return None
        return AuthUser(
            user_id=user_record["user_id"],
            name=user_record.get("name"),
            email=user_record["email"],
            google_id=user_record.get("google_id"),
            created_at=user_record.get("created_at"),
        )


def authenticate_google_user(
    email: str, google_id: str, name: Optional[str] = None
) -> Optional[AuthUser]:
    """Authenticates an existing registered user via verified Google OAuth credentials.

    STRICT ENFORCEMENT: A user who has NOT registered an account in LegalLens AI
    must NEVER be allowed to log in through Continue with Google.
    - If the verified email exists in LegalLens AI user database:
        Links Google subject ID (if needed) and returns AuthUser.
    - If the email does NOT exist in the database:
        Returns None. NEVER creates a user account or session.
    """
    clean_email = (email or "").strip().lower()
    if not clean_email or not google_id or not google_id.strip():
        return None

    clean_google_id = google_id.strip()
    clean_name = (name or "").strip()

    with _auth_lock:
        users = _load_json_file(USERS_JSON)
        # Case-insensitive lookup
        user_record = users.get(clean_email)
        record_key = clean_email
        if not user_record:
            for k, v in users.items():
                if k.lower() == clean_email or (isinstance(v, dict) and v.get("email", "").lower() == clean_email):
                    user_record = v
                    record_key = k
                    break

        if not user_record:
            # STRICT REQUIREMENT: Never create an account for unregistered users
            return None

        # Existing account: link Google subject ID and update name if unset
        modified = False
        if user_record.get("google_id") != clean_google_id:
            user_record["google_id"] = clean_google_id
            modified = True
        if clean_name and not user_record.get("name"):
            user_record["name"] = clean_name
            modified = True

        if modified:
            users[record_key] = user_record
            _save_json_file(USERS_JSON, users)

        return AuthUser(
            user_id=user_record["user_id"],
            name=user_record.get("name"),
            email=user_record["email"],
            google_id=user_record.get("google_id"),
            created_at=user_record.get("created_at"),
        )


def authenticate_or_create_google_user(
    email: str, google_id: str, name: Optional[str] = None
) -> Optional[AuthUser]:
    """Compatibility alias: calls authenticate_google_user.
    Never automatically registers users through Google OAuth.
    """
    return authenticate_google_user(email=email, google_id=google_id, name=name)


def create_session(user: AuthUser) -> str:
    """Creates a secure session token and persists to sessions.json."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_MAX_AGE_SECONDS)

    with _auth_lock:
        sessions = _load_json_file(SESSIONS_JSON)
        sessions[token] = {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "google_id": user.google_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "expires_at": expires_at.isoformat(timespec="seconds"),
        }
        _save_json_file(SESSIONS_JSON, sessions)

    return token


def get_session_user(token: str) -> Optional[AuthUser]:
    """Validates session token and returns AuthUser if active and unexpired."""
    if not token or not isinstance(token, str):
        return None

    with _auth_lock:
        sessions = _load_json_file(SESSIONS_JSON)
        session = sessions.get(token)
        if not session:
            return None

        expires_str = session.get("expires_at")
        if expires_str:
            try:
                expires_at = datetime.fromisoformat(expires_str)
                if expires_at < datetime.now(timezone.utc):
                    sessions.pop(token, None)
                    _save_json_file(SESSIONS_JSON, sessions)
                    return None
            except Exception:
                return None

        return AuthUser(
            user_id=session["user_id"],
            name=session.get("name"),
            email=session["email"],
            google_id=session.get("google_id"),
            created_at=session.get("created_at"),
        )


def invalidate_session(token: str) -> bool:
    """Invalidates and deletes a session token."""
    if not token:
        return False
    with _auth_lock:
        sessions = _load_json_file(SESSIONS_JSON)
        if token in sessions:
            sessions.pop(token, None)
            _save_json_file(SESSIONS_JSON, sessions)
            return True
    return False
