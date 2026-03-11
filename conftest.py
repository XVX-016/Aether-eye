def pytest_ignore_collect(collection_path, config):
    markexpr = getattr(config.option, "markexpr", "") or ""
    if "not torch" in markexpr:
        if collection_path.name in {"test_yolo.py", "test_tiling.py"}:
            return True
    return False
