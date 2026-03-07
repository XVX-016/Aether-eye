## Aether Eye Monorepo

Production-ready monorepo for aircraft intelligence and satellite change detection.

### Structure

- **backend**: FastAPI APIs and inference services.
- **ml-core**: PyTorch/ONNX model pipelines and training/export scripts.
- **frontend**: Next.js UI (home + operations dashboard components).

### Local Start Commands (Windows)

Shortcut scripts (run from repo root):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-backend.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-frontend.ps1
```

If you are already inside `backend`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-backend.ps1
```

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
- Backend startup scripts auto-detect Python at `C:\mlenv\Scripts\python.exe` or `C:\mlenv\venv\Scripts\python.exe`.
- Frontend now uses a built-in Next.js `/api/*` proxy to backend, so you no longer need to set `NEXT_PUBLIC_API_BASE_URL` for local runs.

### Stanford VisionBasedAircraftDAA Integration

Generate 500 synthetic samples from Stanford generator (copies output into `data/raw/stanford_aircraft/...`):

```powershell
cd C:\Computing\Aether-eye
& "C:\mlenv\venv\Scripts\python.exe" .\scripts\generate_stanford_aircraft_data.py `
  --dataset-name stanford_military_500 `
  --craft "King Air C90" `
  --train 450 `
  --valid 50 `
  --location "Palo Alto"
```

Convert Stanford output to YOLOv8 format (`images/`, `labels/`, `data.yaml`):

```powershell
cd C:\Computing\Aether-eye
& "C:\mlenv\venv\Scripts\python.exe" .\scripts\convert_stanford_to_yolov8.py `
  --input-dir .\data\raw\stanford_aircraft\stanford_military_500 `
  --output-dir .\data\processed\stanford_yolo
```

Prepare ImageFolder classification dataset from Stanford metadata:

```powershell
cd C:\Computing\Aether-eye
& "C:\mlenv\venv\Scripts\python.exe" .\scripts\prepare_stanford_classification_dataset.py `
  --input-dir .\data\raw\stanford_aircraft\stanford_military_500 `
  --output-dir .\data\processed\stanford_aircraft_cls
```

Train ViT on generated classification data:

```powershell
cd C:\Computing\Aether-eye
& "C:\mlenv\venv\Scripts\python.exe" .\scripts\train_vit_stanford.py `
  --data-root .\data\processed\stanford_aircraft_cls `
  --output-dir .\experiments\aircraft\stanford_vit `
  --model vit_small_patch16_224 `
  --epochs 20 `
  --batch-size 16
```

Download SpaceNet-7 and DOTA using KaggleHub into change-detection dataset root:

```powershell
cd C:\Computing\Aether-eye
& "C:\mlenv\venv\Scripts\python.exe" .\scripts\download_kaggle_satellite_datasets.py `
  --target-root .\ml-core\DATASET\Satellite-Change
```

Note:
- VisionBasedAircraftDAA generator supports aircraft configured in its `constants.py` (`Cessna Skyhawk`, `Boeing 737-800`, `King Air C90`) unless that repo is extended with additional aircraft definitions.
