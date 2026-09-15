import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat",
}


class _FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = np.array(cls)
        self.conf = np.array(conf)
        self.xyxy = np.array(xyxy)

    def __len__(self):
        return len(self.cls)


class _FakeMasks:
    def __init__(self, polygons):
        self.xy = [np.array(polygon) for polygon in polygons]


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes(cls=[0], conf=[0.87], xyxy=[[10.0, 20.0, 100.0, 150.0]])
        self.masks = _FakeMasks([[[10, 20], [100, 20], [100, 150], [10, 150]]])
        self.names = CLASS_NAMES
        self.orig_shape = (480, 640)

    def plot(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (640, 480), color=(120, 120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    import app.main as main_module
    from app.model_loader import _decode_image

    def fake_run_inference(image, conf=None, iou=None):
        _decode_image(image)
        return [_FakeResult()]

    monkeypatch.setattr(main_module, "get_model", lambda: object())
    monkeypatch.setattr(main_module, "get_weights_path_used", lambda: "models/best.pt")
    monkeypatch.setattr(main_module, "run_inference", fake_run_inference)

    with TestClient(main_module.app) as test_client:
        yield test_client
