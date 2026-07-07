import io
import pytest
from PIL import Image
from gateway.db import Db
from gateway.store import SourceStore, SourceEvicted

def make_store(tmp_path, max_bytes=10**9):
    db = Db(tmp_path / "g.sqlite3")
    return SourceStore(db=db, root=tmp_path / "sources", max_bytes=max_bytes)

def png_bytes(w=8, h=6, color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()

def test_add_returns_record_and_dedups(tmp_path):
    st = make_store(tmp_path)
    r1 = st.add(png_bytes(), "a.png")
    r2 = st.add(png_bytes(), "b.png")   # same bytes -> same id
    assert r1["source_id"] == r2["source_id"]
    assert r1["kind"] == "std"
    assert (r1["width"], r1["height"]) == (8, 6)

def test_sidecars_and_get(tmp_path):
    st = make_store(tmp_path)
    r = st.add(png_bytes(), "a.png", exif={"ISO": 100}, rrdata={"rating": 3})
    rec = st.get(r["source_id"])
    assert rec.exif == {"ISO": 100}
    assert rec.rrdata == {"rating": 3}
    assert rec.path.exists()
    assert rec.kind == "std"

def test_raw_kind_no_dims_uses_client_dims(tmp_path):
    st = make_store(tmp_path)
    r = st.add(b"\x00" * 100, "shot.ARW", client_width=8640, client_height=5760)
    assert r["kind"] == "raw"
    rec = st.get(r["source_id"])
    assert (rec.width, rec.height) == (8640, 5760)

def test_unknown_id_returns_none(tmp_path):
    st = make_store(tmp_path)
    assert st.get("deadbeef" * 8) is None

def test_eviction_raises_gone_on_get(tmp_path):
    st = make_store(tmp_path, max_bytes=100)  # tiny cap: a=77B, b=80B; combined 157B > cap

    r1 = st.add(png_bytes(color=(1, 2, 3)), "a.png")
    st.add(png_bytes(16, 12, (4, 5, 6)), "b.png")  # bigger, evicts a
    with pytest.raises(SourceEvicted):
        st.get(r1["source_id"])
