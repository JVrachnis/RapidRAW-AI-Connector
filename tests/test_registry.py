import pytest
from gateway import registry

@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(registry.REGISTRY)
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)

async def dummy_handler(ctx):
    return {"ok": True}

def test_register_and_describe():
    cap = registry.Capability(
        id="echo", title="Echo", params_schema={"type": "object"}, handler=dummy_handler)
    registry.register(cap)
    assert registry.get("echo") is cap
    desc = registry.describe()
    assert desc == [{"id": "echo", "title": "Echo",
                     "params_schema": {"type": "object"}, "modes": None, "presets": None}]

def test_duplicate_id_rejected():
    cap = registry.Capability(id="x", title="X", params_schema={}, handler=dummy_handler)
    registry.register(cap)
    with pytest.raises(ValueError):
        registry.register(cap)

def test_validate_params_raises_on_schema_violation():
    cap = registry.Capability(
        id="v", title="V",
        params_schema={"type": "object", "required": ["mode"],
                       "properties": {"mode": {"enum": ["a", "b"]}}},
        handler=dummy_handler)
    registry.register(cap)
    registry.validate_params("v", {"mode": "a"})   # ok
    with pytest.raises(registry.ParamsInvalid):
        registry.validate_params("v", {"mode": "zzz"})
