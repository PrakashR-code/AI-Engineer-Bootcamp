from fastapi.testclient import TestClient

from app import app


def test_ping():
    client = TestClient(app)
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
