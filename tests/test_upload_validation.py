import io
import sys
from pathlib import Path

from starlette.datastructures import UploadFile
from fastapi import HTTPException

# Ensure backend is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from backend.app.api.routes.intelligence import _read_upload_image


def test_corrupted_image_returns_invalid_image():
    fake = UploadFile(filename="bad.jpg", file=io.BytesIO(b"not an image"))
    try:
        _read_upload_image(fake)
        assert False, "Expected HTTPException for invalid image"
    except HTTPException as exc:
        detail = exc.detail
        assert isinstance(detail, dict)
        assert detail.get("error") == "INVALID_IMAGE"
