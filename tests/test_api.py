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


def test_detect_returns_expected_shape(client, sample_image_bytes):
    response = client.post(
        "/api/v1/detect",
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


def test_detect_rejects_bad_content_type(client):
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_detect_rejects_corrupt_image(client):
    response = client.post(
        "/api/v1/detect",
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 422
