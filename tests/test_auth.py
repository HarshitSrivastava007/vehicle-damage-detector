def test_register_creates_user_and_returns_key(client):
    response = client.post("/api/v1/auth/register", json={"email": "new@example.com"})
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com"
    assert body["api_key"].startswith("vdd_")
    assert "user_id" in body


def test_register_duplicate_email_conflicts(client):
    client.post("/api/v1/auth/register", json={"email": "dupe@example.com"})
    response = client.post("/api/v1/auth/register", json={"email": "dupe@example.com"})
    assert response.status_code == 409


def test_me_returns_current_user(client, auth_headers):
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "tester@example.com"


def test_list_keys_never_exposes_raw_key(client, auth_headers):
    response = client.get("/api/v1/auth/keys", headers=auth_headers)
    assert response.status_code == 200
    keys = response.json()
    assert len(keys) == 1
    assert "prefix" in keys[0]
    assert "api_key" not in keys[0]
    assert "key_hash" not in keys[0]


def test_create_additional_key(client, auth_headers):
    response = client.post("/api/v1/auth/keys", headers=auth_headers)
    assert response.status_code == 201
    assert response.json()["api_key"].startswith("vdd_")

    keys = client.get("/api/v1/auth/keys", headers=auth_headers).json()
    assert len(keys) == 2


def test_revoke_key_idempotency(client, auth_headers):
    # Revoke a *second* key while authenticating with the first — revoking
    # the key that's actively authenticating would correctly 401 on the
    # next call, which would defeat this idempotency check.
    extra_key_id = client.post("/api/v1/auth/keys", headers=auth_headers).json()["id"]

    first = client.delete(f"/api/v1/auth/keys/{extra_key_id}", headers=auth_headers)
    assert first.status_code == 204

    second = client.delete(f"/api/v1/auth/keys/{extra_key_id}", headers=auth_headers)
    assert second.status_code == 409


def test_account_endpoints_require_auth(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/auth/keys").status_code == 401
    assert client.post("/api/v1/auth/keys").status_code == 401
    assert client.delete("/api/v1/auth/keys/1").status_code == 401
