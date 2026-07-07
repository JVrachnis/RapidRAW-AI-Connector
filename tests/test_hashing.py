from gateway.hashing import content_id, detect_kind

def test_content_id_is_stable_and_content_addressed():
    a = content_id(b"hello")
    assert a == content_id(b"hello")
    assert a != content_id(b"hello!")
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)

def test_detect_kind():
    assert detect_kind("shot.ARW") == "raw"
    assert detect_kind("x.dng") == "raw"
    assert detect_kind("linear.tiff") == "tiff"
    assert detect_kind("linear.TIF") == "tiff"
    assert detect_kind("img.jpg") == "std"
    assert detect_kind("noext") == "std"
