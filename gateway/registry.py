import importlib
import pkgutil
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
import jsonschema

class ParamsInvalid(Exception):
    pass

@dataclass
class Capability:
    id: str
    title: str
    params_schema: dict
    handler: Callable[["JobContext"], Awaitable[dict]]  # noqa: F821 (defined in gateway.jobs)
    modes: Optional[list] = None
    presets: Optional[list] = None

REGISTRY: dict[str, Capability] = {}

def register(cap: Capability) -> None:
    if cap.id in REGISTRY:
        raise ValueError(f"capability {cap.id!r} already registered")
    REGISTRY[cap.id] = cap

def get(cap_id: str) -> Optional[Capability]:
    return REGISTRY.get(cap_id)

def describe(caps: Optional[dict[str, Capability]] = None) -> list[dict]:
    caps = REGISTRY if caps is None else caps
    return [{"id": c.id, "title": c.title, "params_schema": c.params_schema,
             "modes": c.modes, "presets": c.presets} for c in caps.values()]

def validate_params(cap_id: str, params: dict,
                    caps: Optional[dict[str, Capability]] = None) -> None:
    cap = (REGISTRY if caps is None else caps)[cap_id]
    try:
        jsonschema.validate(params, cap.params_schema)
    except jsonschema.ValidationError as e:
        raise ParamsInvalid(e.message) from e

def load_capabilities() -> None:
    """Import every module in the capabilities package; modules call register() at import."""
    import capabilities
    for m in pkgutil.iter_modules(capabilities.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"capabilities.{m.name}")
