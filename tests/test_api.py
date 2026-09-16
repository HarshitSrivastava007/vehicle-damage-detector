def test_index_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Vehicle Damage Detector" in response.text


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_detect_returns_expected_shape(client, sample_image_bytes, auth_headers):
    response = client.post(
        "/api/v1/detect",
        headers=auth_headers,
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "test.jpg"
    assert isinstance(body["detections"], list)
    assert body["count"] == len(body["detections"])
    detection = body["detections"][0]
    assert detection["class_name"] == "dent"
    assert 0.0 <= detection["confidence"] <= 1.0
    assert set(detection["bbox"].keys()) == {"x1", "y1", "x2", "y2"}


def test_detect_rejects_bad_content_type(client, auth_headers):
    response = client.post(
        "/api/v1/detect",
        headers=auth_headers,
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_detect_rejects_corrupt_image(client, auth_headers):
    response = client.post(
        "/api/v1/detect",
        headers=auth_headers,
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 422


def test_detect_requires_api_key(client, sample_image_bytes):
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 401


def test_detect_rejects_revoked_key(client, sample_image_bytes):
    register = client.post("/api/v1/auth/register", json={"email": "revoked@example.com"})
    headers = {"Authorization": f"Bearer {register.json()['api_key']}"}
    key_id = client.get("/api/v1/auth/keys", headers=headers).json()[0]["id"]

    revoke = client.delete(f"/api/v1/auth/keys/{key_id}", headers=headers)
    assert revoke.status_code == 204

    response = client.post(
        "/api/v1/detect",
        headers=headers,
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 401


def test_detection_history_records_after_detect(client, sample_image_bytes, auth_headers):
    detect_response = client.post(
        "/api/v1/detect",
        headers=auth_headers,
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert detect_response.status_code == 200

    history = client.get("/api/v1/detections", headers=auth_headers)
    assert history.status_code == 200
    entries = history.json()
    assert len(entries) == 1
    assert entries[0]["filename"] == "test.jpg"
    assert entries[0]["detection_count"] == detect_response.json()["count"]
