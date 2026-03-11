from types import SimpleNamespace
import pytest

try:
    import pydantic_settings  # noqa: F401
except Exception:
    pytest.skip("pydantic_settings not available", allow_module_level=True)

from app.services.ingestion_service import filter_new_items


def test_filter_new_items_dedup():
    items = [
        SimpleNamespace(id="scene_a"),
        SimpleNamespace(id="scene_b"),
    ]
    existing = {("scene_a", "sentinel-2-l2a")}
    new_items = filter_new_items(items, existing, "sentinel-2-l2a")
    assert len(new_items) == 1
    assert new_items[0].id == "scene_b"
