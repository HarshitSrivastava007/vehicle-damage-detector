from datetime import timedelta

from app.session_auth import hash_session_token, utcnow


def test_session_cookie_authenticates_key_management(session_user):
    # The dashboard SPA only ever has a session cookie, never an API key —
    # key management and /detect must accept session auth too, not just
    # the Authorization: Bearer header used by programmatic API callers.
    response = session_user.post("/api/v1/auth/keys")
    assert response.status_code == 201

    keys = session_user.get("/api/v1/auth/keys").json()
    assert len(keys) == 2  # one from register(), one just created above


def test_session_cookie_authenticates_detect(session_user, sample_image_bytes):
    response = session_user.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200

    history = session_user.get("/api/v1/detections").json()
    assert len(history) == 1


def test_login_success_sets_cookie_and_returns_user_info(client):
    client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "s3cret-password"})
    response = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "s3cret-password"})
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "a@example.com"
    assert body["is_admin"] is False
    assert "vdd_session" in response.cookies


def test_login_wrong_password_generic_401(client):
    client.post("/api/v1/auth/register", json={"email": "b@example.com", "password": "correct-password"})
    response = client.post("/api/v1/auth/login", json={"email": "b@example.com", "password": "wrong-password"})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid email or password"


def test_login_unknown_email_generic_401(client):
    response = client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever"})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid email or password"


def test_login_rejected_for_account_without_password(client):
    # register() without a password — API-key-only account
    client.post("/api/v1/auth/register", json={"email": "apikeyonly@example.com"})
    response = client.post("/api/v1/auth/login", json={"email": "apikeyonly@example.com", "password": "anything"})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid email or password"


def test_logout_clears_session(session_user):
    assert session_user.get("/api/v1/auth/session").status_code == 200

    logout = session_user.post("/api/v1/auth/logout")
    assert logout.status_code == 204

    assert session_user.get("/api/v1/auth/session").status_code == 401


def test_logout_idempotent_without_prior_login(client):
    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 204


def test_dashboard_endpoints_require_session(client):
    assert client.get("/api/v1/auth/session").status_code == 401
    assert client.get("/api/v1/usage/summary").status_code == 401
    assert client.get("/api/v1/admin/users").status_code == 401


def test_admin_endpoints_403_for_non_admin(session_user):
    assert session_user.get("/api/v1/admin/users").status_code == 403
    assert session_user.patch("/api/v1/admin/users/1/cost", json={"cost_per_call": "1.00"}).status_code == 403


def test_admin_can_list_users_with_detection_counts(admin_session_user):
    response = admin_session_user.get("/api/v1/admin/users")
    assert response.status_code == 200
    users = response.json()
    assert len(users) == 1
    assert users[0]["email"] == "dash@example.com"
    assert users[0]["total_detections"] == 0


def test_admin_can_set_user_cost_per_call(admin_session_user):
    user_id = admin_session_user.get("/api/v1/admin/users").json()[0]["id"]
    response = admin_session_user.patch(f"/api/v1/admin/users/{user_id}/cost", json={"cost_per_call": "0.05"})
    assert response.status_code == 200
    assert response.json()["cost_per_call"] == "0.0500"


def test_negative_cost_per_call_rejected(admin_session_user):
    user_id = admin_session_user.get("/api/v1/admin/users").json()[0]["id"]
    response = admin_session_user.patch(f"/api/v1/admin/users/{user_id}/cost", json={"cost_per_call": "-1"})
    assert response.status_code == 422


def test_set_cost_unknown_user_404(admin_session_user):
    response = admin_session_user.patch("/api/v1/admin/users/999999/cost", json={"cost_per_call": "1.00"})
    assert response.status_code == 404


def test_usage_summary_zero_for_new_user_with_no_calls(session_user):
    response = session_user.get("/api/v1/usage/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["calls_all_time"] == 0
    assert body["cost_all_time"] == "0.0000"


def test_detection_log_cost_snapshot_unaffected_by_later_rate_change(client, db_session, sample_image_bytes):
    from app.db_models import User

    # Register captures the API key (used for /detect) directly from the
    # response — the session_user/admin_session_user fixtures discard it,
    # since /api/v1/auth/keys itself requires API-key auth, not a session
    # cookie, so it can't be used to mint a key for an already-logged-in
    # session client.
    register = client.post(
        "/api/v1/auth/register",
        json={"email": "billed@example.com", "password": "pw"},
    )
    api_headers = {"Authorization": f"Bearer {register.json()['api_key']}"}

    user = db_session.query(User).filter_by(email="billed@example.com").one()
    user.is_admin = True
    db_session.commit()

    client.post("/api/v1/auth/login", json={"email": "billed@example.com", "password": "pw"})

    client.patch(f"/api/v1/admin/users/{user.id}/cost", json={"cost_per_call": "1.5"})

    detect1 = client.post(
        "/api/v1/detect",
        headers=api_headers,
        files={"file": ("a.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert detect1.status_code == 200

    client.patch(f"/api/v1/admin/users/{user.id}/cost", json={"cost_per_call": "3.0"})

    detect2 = client.post(
        "/api/v1/detect",
        headers=api_headers,
        files={"file": ("b.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert detect2.status_code == 200

    history = client.get("/api/v1/detections", headers=api_headers).json()
    costs_by_filename = {row["filename"]: row["cost"] for row in history}
    assert costs_by_filename["a.jpg"] == "1.5000"
    assert costs_by_filename["b.jpg"] == "3.0000"

    summary = client.get("/api/v1/usage/summary").json()
    assert summary["cost_all_time"] == "4.5000"


def test_expired_session_rejected(client, db_session):
    from app.db_models import User, UserSession

    client.post("/api/v1/auth/register", json={"email": "expired@example.com", "password": "pw"})
    user = db_session.query(User).filter_by(email="expired@example.com").one()

    raw_token = "test-expired-token"
    db_session.add(
        UserSession(
            user_id=user.id,
            session_token_hash=hash_session_token(raw_token),
            expires_at=utcnow() - timedelta(days=1),
        )
    )
    db_session.commit()

    client.cookies.set("vdd_session", raw_token)
    response = client.get("/api/v1/auth/session")
    assert response.status_code == 401


def test_revoked_session_rejected(client, db_session):
    from app.db_models import User, UserSession

    client.post("/api/v1/auth/register", json={"email": "revoked@example.com", "password": "pw"})
    user = db_session.query(User).filter_by(email="revoked@example.com").one()

    raw_token = "test-revoked-token"
    db_session.add(
        UserSession(
            user_id=user.id,
            session_token_hash=hash_session_token(raw_token),
            expires_at=utcnow() + timedelta(days=1),
            revoked_at=utcnow(),
        )
    )
    db_session.commit()

    client.cookies.set("vdd_session", raw_token)
    response = client.get("/api/v1/auth/session")
    assert response.status_code == 401


def test_admin_can_create_user(admin_session_user):
    response = admin_session_user.post(
        "/api/v1/admin/users",
        json={"email": "newbie@example.com", "password": "s3cret-password", "cost_per_call": "2.5"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "newbie@example.com"
    assert body["is_admin"] is False
    assert body["is_active"] is True
    assert body["cost_per_call"] == "2.5000"
    assert body["api_key"].startswith("vdd_")

    users = admin_session_user.get("/api/v1/admin/users").json()
    assert len(users) == 2


def test_admin_created_user_can_log_in_and_use_api_key(admin_session_user, client):
    create = admin_session_user.post(
        "/api/v1/admin/users",
        json={"email": "newbie2@example.com", "password": "s3cret-password"},
    )
    api_key = create.json()["api_key"]

    login = client.post("/api/v1/auth/login", json={"email": "newbie2@example.com", "password": "s3cret-password"})
    assert login.status_code == 200

    headers = {"Authorization": f"Bearer {api_key}"}
    assert client.get("/api/v1/auth/keys", headers=headers).status_code == 200


def test_admin_create_user_duplicate_email_conflicts(admin_session_user):
    admin_session_user.post("/api/v1/admin/users", json={"email": "dupe2@example.com", "password": "pw"})
    response = admin_session_user.post("/api/v1/admin/users", json={"email": "dupe2@example.com", "password": "pw"})
    assert response.status_code == 409


def test_create_user_requires_admin(session_user):
    response = session_user.post("/api/v1/admin/users", json={"email": "x@example.com", "password": "pw"})
    assert response.status_code == 403


def test_admin_can_deactivate_and_reactivate_user(admin_session_user):
    from fastapi.testclient import TestClient

    # Use a second, independent TestClient (own cookie jar, same app/DB) for
    # the target user's login/API-key checks — reusing admin_session_user's
    # own client would overwrite its session cookie with the target user's,
    # breaking the subsequent admin PATCH calls in this same test.
    other_client = TestClient(admin_session_user.app)

    create = admin_session_user.post(
        "/api/v1/admin/users",
        json={"email": "togglee@example.com", "password": "s3cret-password"},
    )
    user_id = create.json()["id"]
    api_key = create.json()["api_key"]
    headers = {"Authorization": f"Bearer {api_key}"}

    # active: login and API key both work
    assert other_client.post(
        "/api/v1/auth/login", json={"email": "togglee@example.com", "password": "s3cret-password"}
    ).status_code == 200
    assert other_client.get("/api/v1/auth/keys", headers=headers).status_code == 200

    deactivate = admin_session_user.patch(f"/api/v1/admin/users/{user_id}/status", json={"is_active": False})
    assert deactivate.status_code == 200
    assert deactivate.json()["is_active"] is False

    # inactive: existing API key immediately stops working (no need to revoke it explicitly)
    assert other_client.get("/api/v1/auth/keys", headers=headers).status_code == 401
    # inactive: fresh login attempt is rejected with a specific message
    login_attempt = other_client.post(
        "/api/v1/auth/login", json={"email": "togglee@example.com", "password": "s3cret-password"}
    )
    assert login_attempt.status_code == 403
    assert login_attempt.json()["detail"] == "account is inactive"

    reactivate = admin_session_user.patch(f"/api/v1/admin/users/{user_id}/status", json={"is_active": True})
    assert reactivate.status_code == 200
    assert reactivate.json()["is_active"] is True

    # reactivated: the same (never-revoked) API key works again immediately
    assert other_client.get("/api/v1/auth/keys", headers=headers).status_code == 200


def test_deactivating_existing_session_blocks_it_immediately(admin_session_user):
    from fastapi.testclient import TestClient

    # Deactivating the admin's own account is blocked by the self-lockout
    # guard (tested separately), so create a second user, log them in on
    # their own TestClient (independent cookie jar, same app/DB), then
    # confirm their still-valid session cookie stops working the moment
    # an admin deactivates them — not just on their next login attempt.
    create = admin_session_user.post(
        "/api/v1/admin/users", json={"email": "livesession@example.com", "password": "s3cret-password"}
    )
    user_id = create.json()["id"]

    other_client = TestClient(admin_session_user.app)
    login = other_client.post(
        "/api/v1/auth/login", json={"email": "livesession@example.com", "password": "s3cret-password"}
    )
    assert login.status_code == 200
    assert other_client.get("/api/v1/auth/session").status_code == 200

    admin_session_user.patch(f"/api/v1/admin/users/{user_id}/status", json={"is_active": False})

    assert other_client.get("/api/v1/auth/session").status_code == 401


def test_cannot_deactivate_own_account(admin_session_user):
    me = admin_session_user.get("/api/v1/admin/users").json()
    own_id = next(u["id"] for u in me if u["email"] == "dash@example.com")
    response = admin_session_user.patch(f"/api/v1/admin/users/{own_id}/status", json={"is_active": False})
    assert response.status_code == 400


def test_update_status_unknown_user_404(admin_session_user):
    response = admin_session_user.patch("/api/v1/admin/users/999999/status", json={"is_active": False})
    assert response.status_code == 404
