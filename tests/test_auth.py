import pytest
from fastapi.testclient import TestClient
from engine import Settings
from gateway.app import create_app

@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    settings = Settings(CACHE_DIR=tmp_path, GATEWAY_TOKEN="sekrit")
    app = create_app(settings, load_caps=False)
    with TestClient(app) as c:
        yield c

def test_rejects_missing_token(auth_client):
    assert auth_client.get("/capabilities").status_code == 401

def test_accepts_bearer_token(auth_client):
    r = auth_client.get("/capabilities", headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200

def test_health_is_open(auth_client):
    assert auth_client.get("/health").status_code == 200
