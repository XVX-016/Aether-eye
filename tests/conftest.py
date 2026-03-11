import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def pytest_ignore_collect(collection_path, config):
    markexpr = getattr(config.option, "markexpr", "") or ""
    if "not torch" in markexpr:
        if collection_path.name in {"test_yolo.py", "test_tiling.py"}:
            return True
    return False
