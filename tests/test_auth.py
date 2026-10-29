"""Authentication and session management tests for LegalLens AI.

Tests:
1. Registration with name, email, and password
2. Registration validation (empty name, invalid email, short password, duplicate email)
3. Password hashing security (PBKDF2 with unique salts, constant-time verification)
4. Successful login and session token creation
5. Rejection on incorrect password or non-existent user
6. Google OAuth user creation and existing account linking without duplication
7. Google OAuth URL endpoint (configured vs unconfigured setup instructions)
8. Google OAuth callback error handling
9. Session persistence, me endpoint, and logout
10. Route protection and redirects (/ and /login)
"""
import pytest
from fastapi.testclient import TestClient

from app import auth, config, main
from app.auth import hash_password, verify_password


@pytest.fixture(autouse=True)
def clean_auth_storage(tmp_path, monkeypatch):
    """Isolates user and session storage to a fresh temporary directory per test."""
    users_file = tmp_path / "users.json"
    sessions_file = tmp_path / "sessions.json"
    monkeypatch.setattr(config, "USERS_JSON", users_file)
    monkeypatch.setattr(config, "SESSIONS_JSON", sessions_file)
    monkeypatch.setattr(auth, "USERS_JSON", users_file)
    monkeypatch.setattr(auth, "SESSIONS_JSON", sessions_file)
    yield


def test_password_hashing_and_verification():
    salt_hex, hash_hex = hash_password("SuperSecret123!")
    assert len(salt_hex) == 32  # 16 bytes = 32 hex chars
    assert len(hash_hex) == 64  # sha256 = 32 bytes = 64 hex chars

    # Password matches
    assert verify_password("SuperSecret123!", salt_hex, hash_hex) is True
    # Wrong password fails
    assert verify_password("WrongPassword!", salt_hex, hash_hex) is False

    # Different salts yield different hashes for same password
    salt2, hash2 = hash_password("SuperSecret123!")
    assert salt_hex != salt2
    assert hash_hex != hash2


def test_registration_success():
    client = TestClient(main.app)
    res = client.post(
        "/api/auth/register",
        json={
            "name": "Adv. Sarah Jenkins",
            "email": "counsel@legallens.ai",
            "password": "SecurePassword123!",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "registered"
    assert "user" in data
    assert data["user"]["name"] == "Adv. Sarah Jenkins"
    assert data["user"]["email"] == "counsel@legallens.ai"
    assert "user_id" in data["user"]
    # Password verification is NOT bypassed automatically: session cookie is not set
    assert config.SESSION_COOKIE_NAME not in res.cookies


def test_registration_validation_errors():
    client = TestClient(main.app)

    # Empty name
    res_name = client.post(
        "/api/auth/register",
        json={"name": "   ", "email": "valid@court.gov", "password": "SecurePassword123!"},
    )
    assert res_name.status_code in (400, 422)

    # Invalid email format
    res_email = client.post(
        "/api/auth/register",
        json={"name": "Counsel", "email": "not-an-email", "password": "SecurePassword123!"},
    )
    assert res_email.status_code == 400
    assert "valid email" in res_email.json()["detail"].lower()

    # Short password
    res_pw = client.post(
        "/api/auth/register",
        json={"name": "Counsel", "email": "counsel@legallens.ai", "password": "short"},
    )
    assert res_pw.status_code in (400, 422)

    # Duplicate registration
    client.post(
        "/api/auth/register",
        json={"name": "First User", "email": "counsel@legallens.ai", "password": "SecurePassword123!"},
    )
    res_dup = client.post(
        "/api/auth/register",
        json={"name": "Second User", "email": "counsel@legallens.ai", "password": "SecurePassword123!"},
    )
    assert res_dup.status_code == 400
    assert "already exists" in res_dup.json()["detail"].lower()

    # Case-insensitive duplicate registration check
    res_dup_case = client.post(
        "/api/auth/register",
        json={"name": "Third User", "email": "COUNSEL@LEGALLENS.AI", "password": "SecurePassword123!"},
    )
    assert res_dup_case.status_code == 400
    assert "already exists" in res_dup_case.json()["detail"].lower()


def test_login_success():
    client = TestClient(main.app)
    # Register first
    client.post(
        "/api/auth/register",
        json={"name": "Attorney", "email": "attorney@court.gov", "password": "StrongPassword99"},
    )

    # Login
    res = client.post(
        "/api/auth/login",
        json={"email": "attorney@court.gov", "password": "StrongPassword99"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["user"]["email"] == "attorney@court.gov"
    assert data["user"]["name"] == "Attorney"
    assert config.SESSION_COOKIE_NAME in res.cookies


def test_login_invalid_credentials():
    client = TestClient(main.app)
    client.post(
        "/api/auth/register",
        json={"name": "Attorney", "email": "attorney@court.gov", "password": "StrongPassword99"},
    )

    # Wrong password
    res = client.post(
        "/api/auth/login",
        json={"email": "attorney@court.gov", "password": "IncorrectPassword"},
    )
    assert res.status_code == 401
    assert "invalid email or password" in res.json()["detail"].lower()

    # Non-existent email
    res2 = client.post(
        "/api/auth/login",
        json={"email": "nonexistent@court.gov", "password": "StrongPassword99"},
    )
    assert res2.status_code == 401
    assert "invalid email or password" in res2.json()["detail"].lower()


def test_login_empty_or_whitespace_credentials():
    client = TestClient(main.app)
    client.post(
        "/api/auth/register",
        json={"name": "Counsel", "email": "counsel@court.gov", "password": "ValidPassword123!"},
    )

    # Empty email
    res_empty_email = client.post(
        "/api/auth/login",
        json={"email": "", "password": "ValidPassword123!"},
    )
    assert res_empty_email.status_code == 401
    assert "invalid email or password" in res_empty_email.json()["detail"].lower()
    assert config.SESSION_COOKIE_NAME not in res_empty_email.cookies

    # Whitespace email
    res_ws_email = client.post(
        "/api/auth/login",
        json={"email": "   ", "password": "ValidPassword123!"},
    )
    assert res_ws_email.status_code == 401
    assert "invalid email or password" in res_ws_email.json()["detail"].lower()
    assert config.SESSION_COOKIE_NAME not in res_ws_email.cookies

    # Empty password
    res_empty_pw = client.post(
        "/api/auth/login",
        json={"email": "counsel@court.gov", "password": ""},
    )
    assert res_empty_pw.status_code == 401
    assert "invalid email or password" in res_empty_pw.json()["detail"].lower()
    assert config.SESSION_COOKIE_NAME not in res_empty_pw.cookies

    # Both empty
    res_both_empty = client.post(
        "/api/auth/login",
        json={"email": "", "password": ""},
    )
    assert res_both_empty.status_code == 401
    assert "invalid email or password" in res_both_empty.json()["detail"].lower()
    assert config.SESSION_COOKIE_NAME not in res_both_empty.cookies

    # Missing fields
    res_missing = client.post(
        "/api/auth/login",
        json={},
    )
    assert res_missing.status_code == 401
    assert "invalid email or password" in res_missing.json()["detail"].lower()
    assert config.SESSION_COOKIE_NAME not in res_missing.cookies


def test_login_establishes_authentication_correctly():
    client = TestClient(main.app)
    client.post(
        "/api/auth/register",
        json={"name": "Justice Brown", "email": "justice.brown@chambers.gov", "password": "SecurePassword2026!"},
    )

    # Correct email + correct password = login succeeds
    login_res = client.post(
        "/api/auth/login",
        json={"email": "justice.brown@chambers.gov", "password": "SecurePassword2026!"},
    )
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert login_data["user"]["email"] == "justice.brown@chambers.gov"
    assert login_data["user"]["name"] == "Justice Brown"
    assert "token" in login_data
    token = login_data["token"]
    assert config.SESSION_COOKIE_NAME in login_res.cookies
    cookie_token = login_res.cookies[config.SESSION_COOKIE_NAME]

    # Verify session persists in backend sessions.json
    session_user = auth.get_session_user(token)
    assert session_user is not None
    assert session_user.email == "justice.brown@chambers.gov"

    # Authenticated client with cookie can access /api/auth/me
    authed_client = TestClient(main.app)
    authed_client.cookies.set(config.SESSION_COOKIE_NAME, cookie_token)
    me_res = authed_client.get("/api/auth/me")
    assert me_res.status_code == 200
    assert me_res.json()["authenticated"] is True
    assert me_res.json()["user"]["email"] == "justice.brown@chambers.gov"

    # Authenticated client visiting /login gets redirected to /
    login_redir = authed_client.get("/login", follow_redirects=False)
    assert login_redir.status_code == 302
    assert login_redir.headers["location"] == "/"

    # Authenticated client visiting / gets dashboard (status 200)
    root_res = authed_client.get("/", follow_redirects=False)
    assert root_res.status_code == 200


def test_login_case_insensitive_email():
    client = TestClient(main.app)
    client.post(
        "/api/auth/register",
        json={"name": "Advocate", "email": "advocate.senior@bar.org", "password": "PasswordBar2026!"},
    )

    # Login using uppercase email
    res = client.post(
        "/api/auth/login",
        json={"email": "ADVOCATE.SENIOR@BAR.ORG", "password": "PasswordBar2026!"},
    )
    assert res.status_code == 200
    assert res.json()["user"]["email"] == "advocate.senior@bar.org"
    assert config.SESSION_COOKIE_NAME in res.cookies


def test_login_migrates_legacy_plaintext_password():
    import json
    client = TestClient(main.app)
    # Simulate a legacy user record stored with a plaintext password
    legacy_users = {
        "legacy.user@court.gov": {
            "user_id": "legacy123",
            "name": "Legacy User",
            "email": "legacy.user@court.gov",
            "password": "PlaintextPassword2026!",
            "salt": None,
            "password_hash": None,
            "google_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    }
    config.USERS_JSON.write_text(json.dumps(legacy_users), encoding="utf-8")

    # Incorrect password fails
    fail_res = client.post(
        "/api/auth/login",
        json={"email": "legacy.user@court.gov", "password": "WrongPassword"},
    )
    assert fail_res.status_code == 401
    assert "invalid email or password" in fail_res.json()["detail"].lower()

    # Correct plaintext password succeeds and triggers migration
    succ_res = client.post(
        "/api/auth/login",
        json={"email": "legacy.user@court.gov", "password": "PlaintextPassword2026!"},
    )
    assert succ_res.status_code == 200
    assert succ_res.json()["user"]["email"] == "legacy.user@court.gov"
    assert config.SESSION_COOKIE_NAME in succ_res.cookies

    # Verify that the database no longer contains plaintext password and now has salt and hash
    updated_users = json.loads(config.USERS_JSON.read_text(encoding="utf-8"))
    rec = updated_users["legacy.user@court.gov"]
    assert "password" not in rec or rec["password"] is None
    assert rec["salt"] is not None and len(rec["salt"]) == 32
    assert rec["password_hash"] is not None and len(rec["password_hash"]) == 64
    assert auth.verify_password("PlaintextPassword2026!", rec["salt"], rec["password_hash"]) is True

    # Subsequent login works via newly stored hash
    succ_res2 = client.post(
        "/api/auth/login",
        json={"email": "legacy.user@court.gov", "password": "PlaintextPassword2026!"},
    )
    assert succ_res2.status_code == 200


def test_google_auth_registration_enforcement_and_linking():
    # 1. Unregistered user attempting Google OAuth must be REJECTED (never auto-created)
    unregistered = auth.authenticate_google_user(
        email="unregistered.counsel@highcourt.gov",
        google_id="google-sub-12345",
        name="Hon. Judge Smith",
    )
    assert unregistered is None
    # Verify user was NOT created in database
    assert auth.get_user_by_email("unregistered.counsel@highcourt.gov") is None

    # Compatibility alias must also reject unregistered users
    alias_res = auth.authenticate_or_create_google_user(
        email="unregistered.counsel@highcourt.gov",
        google_id="google-sub-12345",
    )
    assert alias_res is None

    # 2. Register an account properly first:
    local_user = auth.register_user(
        name="Advocate Lee",
        email="advocate.lee@chambers.com",
        password="MyPassword2026!",
    )
    assert local_user.email == "advocate.lee@chambers.com"

    # 3. Now authenticating via Google with the registered email succeeds (case-insensitive)
    linked_user = auth.authenticate_google_user(
        email="ADVOCATE.LEE@CHAMBERS.COM",
        google_id="google-sub-99999",
        name="Advocate Lee",
    )
    assert linked_user is not None
    assert linked_user.email == "advocate.lee@chambers.com"
    assert linked_user.google_id == "google-sub-99999"

    # Session can be created for the registered and linked user
    token = auth.create_session(linked_user)
    session_user = auth.get_session_user(token)
    assert session_user is not None
    assert session_user.email == "advocate.lee@chambers.com"

    # Existing password authentication still works on the linked account
    pw_auth_linked = auth.authenticate_user("advocate.lee@chambers.com", "MyPassword2026!")
    assert pw_auth_linked is not None
    assert pw_auth_linked.email == "advocate.lee@chambers.com"
    assert pw_auth_linked.google_id == "google-sub-99999"


def test_google_auth_url_endpoint(monkeypatch):
    client = TestClient(main.app)

    # 1. When both credentials unconfigured
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "")

    res_un = client.get("/api/auth/google/url")
    assert res_un.status_code == 200
    assert "no-store" in res_un.headers.get("cache-control", "")
    data_un = res_un.json()
    assert data_un["configured"] is False
    assert data_un["auth_url"] is None
    assert "message" in data_un
    assert "not configured" in data_un["message"].lower()
    assert "setup_instructions" not in data_un
    assert "client_secret" not in data_un

    # 2. When only Client ID is provided, but Secret is missing
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "mock-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "mock-client-id.apps.googleusercontent.com")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "")

    res_partial_id = client.get("/api/auth/google/url")
    assert res_partial_id.status_code == 200
    assert res_partial_id.json()["configured"] is False

    # 3. When only Client Secret is provided, but ID is missing
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "mock-secret")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "mock-secret")

    res_partial_sec = client.get("/api/auth/google/url")
    assert res_partial_sec.status_code == 200
    assert res_partial_sec.json()["configured"] is False

    # 4. When both credentials properly configured
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "mock-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "mock-client-secret")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "mock-client-id.apps.googleusercontent.com")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "mock-client-secret")

    res_configured = client.get("/api/auth/google/url")
    assert res_configured.status_code == 200
    assert "no-store" in res_configured.headers.get("cache-control", "")
    data_conf = res_configured.json()
    assert data_conf["configured"] is True
    assert "accounts.google.com" in data_conf["auth_url"]
    assert "mock-client-id" in data_conf["auth_url"]
    assert "client_secret" not in data_conf
    assert "mock-client-secret" not in res_configured.text


def test_google_auth_callback_errors():
    client = TestClient(main.app)

    # Callback with Google error parameter redirects to /login?error=...
    res_err = client.get("/api/auth/google/callback?error=access_denied", follow_redirects=False)
    assert res_err.status_code == 302
    assert "/login?error=" in res_err.headers["location"]

    # Callback with missing code
    res_no_code = client.get("/api/auth/google/callback", follow_redirects=False)
    assert res_no_code.status_code == 302
    assert "/login?error=" in res_no_code.headers["location"]


def test_google_auth_callback_registration_enforcement(monkeypatch):
    client = TestClient(main.app)

    # Configure credentials in environment and config
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "mock-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "mock-client-secret")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "mock-client-id.apps.googleusercontent.com")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "mock-client-secret")

    state_holder = {"email": "unregistered@domain.com"}

    class MockResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data

        def json(self):
            return self._json_data

    class MockHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, data=None, **kwargs):
            return MockResponse(200, {"access_token": "mock-token-123"})

        def get(self, url, headers=None, **kwargs):
            return MockResponse(200, {
                "sub": "google-sub-777",
                "email": state_holder["email"],
                "name": "Google User",
                "email_verified": True,
            })

    monkeypatch.setattr(main.httpx, "Client", MockHttpClient)

    # 1. Unregistered user attempts Google OAuth callback
    res_unreg = client.get("/api/auth/google/callback?code=test-code", follow_redirects=False)

    # Must redirect to /login with exact error message
    assert res_unreg.status_code == 302
    assert "/login?error=" in res_unreg.headers["location"]
    assert "Account%20not%20found.%20Please%20create%20an%20account%20first." in res_unreg.headers["location"]

    # Must NOT create session or authentication cookie
    assert main.SESSION_COOKIE_NAME not in res_unreg.cookies

    # Must NOT create user account in the database
    assert auth.get_user_by_email("unregistered@domain.com") is None

    # Unregistered user cannot access protected dashboard route (redirected to /login)
    unauthed_root = client.get("/", follow_redirects=False)
    assert unauthed_root.status_code == 302
    assert unauthed_root.headers["location"] == "/login"

    # 2. Register account properly in LegalLens AI
    auth.register_user(
        name="Registered Counsel",
        email="counsel@chambers.com",
        password="ValidPassword2026!",
    )

    # Google userinfo returns email with uppercase casing (case-insensitive test)
    state_holder["email"] = "COUNSEL@CHAMBERS.COM"

    # Registered user attempts Google OAuth callback
    res_reg = client.get("/api/auth/google/callback?code=test-code", follow_redirects=False)

    # Must redirect to workspace /
    assert res_reg.status_code == 302
    assert res_reg.headers["location"] == "/"

    # Must create session and authentication cookie
    assert main.SESSION_COOKIE_NAME in res_reg.cookies
    session_cookie = res_reg.cookies[main.SESSION_COOKIE_NAME]

    # Google sub ID is linked to the existing user
    linked_db_user = auth.get_user_by_email("counsel@chambers.com")
    assert linked_db_user is not None
    assert linked_db_user.google_id == "google-sub-777"

    # Protected endpoint /api/auth/me confirms session authentication
    authed_client = TestClient(main.app)
    authed_client.cookies.set(main.SESSION_COOKIE_NAME, session_cookie)
    me_res = authed_client.get("/api/auth/me")
    assert me_res.status_code == 200
    assert me_res.json()["authenticated"] is True
    assert me_res.json()["user"]["email"] == "counsel@chambers.com"

    # Protected root workspace allows access
    workspace_res = authed_client.get("/", follow_redirects=False)
    assert workspace_res.status_code == 200


def test_auth_me_and_logout():
    client = TestClient(main.app)
    # Register and then login
    client.post(
        "/api/auth/register",
        json={"name": "Partner", "email": "partner@lawfirm.com", "password": "Password2026!"},
    )
    login_res = client.post(
        "/api/auth/login",
        json={"email": "partner@lawfirm.com", "password": "Password2026!"},
    )
    token = login_res.json()["token"]

    # When authenticated via cookie
    me_res = client.get("/api/auth/me")
    assert me_res.status_code == 200
    assert me_res.json()["authenticated"] is True
    assert me_res.json()["user"]["email"] == "partner@lawfirm.com"
    assert me_res.json()["user"]["name"] == "Partner"

    # Also works with Bearer header
    unauthed_client = TestClient(main.app)
    me_header = unauthed_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_header.status_code == 200
    assert me_header.json()["authenticated"] is True

    # Logout
    logout_res = client.post("/api/auth/logout")
    assert logout_res.status_code == 200
    assert logout_res.json()["status"] == "logged_out"

    # Now /api/auth/me shows unauthenticated
    after_logout = client.get("/api/auth/me")
    assert after_logout.status_code == 200
    assert after_logout.json()["authenticated"] is False


def test_route_protection_and_redirects():
    client = TestClient(main.app)

    # 1. Unauthenticated visiting root "/" gets redirected to "/login"
    res_root = client.get("/", follow_redirects=False)
    assert res_root.status_code == 302
    assert res_root.headers["location"] == "/login"

    # 2. Unauthenticated visiting "/login" gets the login HTML page
    res_login = client.get("/login")
    assert res_login.status_code == 200
    assert "LegalLens" in res_login.text
    assert "Continue with Google" in res_login.text
    assert "Sign In" in res_login.text
    assert "Create Account" in res_login.text

    # 3. Logged in user visiting "/" gets the index.html page
    client.post(
        "/api/auth/register",
        json={"name": "Judge", "email": "judge@highcourt.gov", "password": "Password2026!"},
    )
    client.post(
        "/api/auth/login",
        json={"email": "judge@highcourt.gov", "password": "Password2026!"},
    )
    res_authed_root = client.get("/", follow_redirects=False)
    assert res_authed_root.status_code == 200
    assert "Case Workspace" in res_authed_root.text

    # 4. Logged in user visiting "/login" gets redirected to "/"
    res_login_redirect = client.get("/login", follow_redirects=False)
    assert res_login_redirect.status_code == 302
    assert res_login_redirect.headers["location"] == "/"
