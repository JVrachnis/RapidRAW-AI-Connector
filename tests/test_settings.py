from engine import Settings

def test_gateway_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    s = Settings()
    assert s.GATEWAY_TOKEN is None
    assert s.GATEWAY_CACHE_MAX_GB == 20
    assert s.GATEWAY_RESULT_TTL_HOURS == 24
    assert s.GATEWAY_JOB_TIMEOUT_S == 600
    assert s.GATEWAY_TOOLS_DIR.endswith("comfy")
    assert str(s.gateway_db_path).endswith("gateway.sqlite3")

def test_gateway_settings_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKEN", "sekrit")
    monkeypatch.setenv("GATEWAY_JOB_TIMEOUT_S", "5")
    monkeypatch.setenv("GATEWAY_DB_PATH", str(tmp_path / "j.db"))
    s = Settings()
    assert s.GATEWAY_TOKEN == "sekrit"
    assert s.GATEWAY_JOB_TIMEOUT_S == 5
    assert s.gateway_db_path == tmp_path / "j.db"
