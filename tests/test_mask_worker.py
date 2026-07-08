"""Pure-logic unit tests for the persistent mask worker (Phase 2).

These import capabilities.mask_tools.mask_worker WITHOUT torch/cv2/pelib present:
the module guards every heavy import inside functions, so its LRU + detection
cache keying classes are unit-testable on a CPU-only box. The GPU parity paths
are validated separately by the live parity gate.
"""
import numpy as np
from capabilities.mask_tools import mask_worker as W


def test_module_imports_without_torch():
    # If this import (at module top) pulled in torch/cv2/pelib, this test file
    # would fail to collect on a GPU-free box. Reaching here proves the guard.
    assert hasattr(W, "SourceLRU")
    assert hasattr(W, "SourceState")
    assert hasattr(W, "detection_key")


def test_detection_key_stable_and_threshold_rounded():
    k1 = W.detection_key("bike", "det", 0.5, 0.5)
    k2 = W.detection_key("bike", "det", 0.50000001, 0.49999999)
    assert k1 == k2                       # float jitter collapses to one key
    assert k1 != W.detection_key("rider", "det", 0.5, 0.5)
    assert k1 != W.detection_key("bike", "chan-B", 0.5, 0.5)
    assert k1 != W.detection_key("bike", "det", 0.6, 0.5)


def test_source_state_detection_cache_roundtrip():
    st = W.SourceState("s1")
    assert st.get_detections("bike", "det", 0.5, 0.5) is None
    val = [(np.ones((4, 4), bool), 0.9)]
    st.put_detections("bike", "det", 0.5, 0.5, val)
    got = st.get_detections("bike", "det", 0.5, 0.5)
    assert got is val
    # Different concept/rep/threshold miss.
    assert st.get_detections("rider", "det", 0.5, 0.5) is None
    assert st.get_detections("bike", "det", 0.7, 0.5) is None


def test_lru_get_or_create_and_reuse():
    lru = W.SourceLRU(cap=2)
    a = lru.get_or_create("a")
    assert lru.get_or_create("a") is a    # same object reused
    assert len(lru) == 1
    assert lru.ids() == ["a"]


def test_lru_evicts_least_recently_used_beyond_cap():
    lru = W.SourceLRU(cap=2)
    lru.get_or_create("a")
    lru.get_or_create("b")
    lru.get_or_create("c")                # evicts "a" (LRU)
    assert "a" not in lru
    assert set(lru.ids()) == {"b", "c"}
    assert len(lru) == 2


def test_lru_access_refreshes_recency():
    lru = W.SourceLRU(cap=2)
    a = lru.get_or_create("a")
    lru.get_or_create("b")
    assert lru.get("a") is a              # touch "a" -> now MRU, "b" is LRU
    lru.get_or_create("c")               # evicts "b", keeps "a"
    assert "b" not in lru
    assert set(lru.ids()) == {"a", "c"}


def test_lru_drop_removes_state():
    lru = W.SourceLRU(cap=2)
    lru.get_or_create("a")
    lru.drop("a")
    assert "a" not in lru
    assert len(lru) == 0
    lru.drop("nonexistent")              # no error


def test_is_oom_detects_variants():
    class OutOfMemoryError(Exception):
        pass
    assert W._is_oom(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert W._is_oom(OutOfMemoryError("boom"))
    assert not W._is_oom(RuntimeError("cannot read image"))
