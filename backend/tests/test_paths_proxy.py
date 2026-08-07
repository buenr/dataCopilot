import pytest
from app.paths import UnsafePath, safe_path
from app.proxy import parse_preview_path, preview_target


def test_path_safety_rejects_traversal(tmp_path):
    with pytest.raises(UnsafePath):
        safe_path(tmp_path, "../secret")
    with pytest.raises(UnsafePath):
        safe_path(tmp_path, "/etc/passwd")


def test_preview_path_preserves_nested_path_and_query():
    assert parse_preview_path("8501", "assets/app.js") == (8501, "/assets/app.js")
    assert preview_target(43123, "/_stcore/stream", "x=1") == "http://127.0.0.1:43123/_stcore/stream?x=1"
