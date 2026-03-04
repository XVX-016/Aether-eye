## Aether Eye Monorepo

Production-ready monorepo for aircraft intelligence and satellite change detection.

### Structure

- **backend**: FastAPI APIs and inference services.
- **ml-core**: PyTorch/ONNX model pipelines and training/export scripts.
- **frontend**: Next.js UI (home + operations dashboard components).

### Local Start Commands (Windows)

Backend (FastAPI):

```powershell
cd C:\Computing\Aether-eye\backend
$env:PYTHONPATH="C:\Computing\Aether-eye\ml-core"
$env:BACKEND_PYTHON_EXECUTABLE="C:\mlenv\Scripts\python.exe"
& "C:\mlenv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend (Next.js):

```powershell
cd C:\Computing\Aether-eye\frontend
npm install
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000/api"
npm run dev
```

Production frontend run:

```powershell
cd C:\Computing\Aether-eye\frontend
npm run build
npm run start
```

### Notes

- If backend import fails for `aether_ml`, verify `PYTHONPATH` points to `C:\Computing\Aether-eye\ml-core`.
- Backend startup enforces `BACKEND_PYTHON_EXECUTABLE`; keep it aligned with the interpreter you use.
