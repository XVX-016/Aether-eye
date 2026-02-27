## Aether Eye Monorepo

Production-ready monorepo skeleton for an AI-based aerial recognition and satellite change detection system.

### Structure

- **backend**: FastAPI application exposing HTTP APIs and orchestrating inference.
- **ml-core**: PyTorch-based ML core packaged as a standalone library with models, preprocessing, and inference utilities.
- **frontend**: React SPA for visualization, annotations, and change-detection workflows.
- **docker-compose.yml**: Orchestration for API and frontend services.

### High-level Architecture

- **ML core isolation**: All model definitions, preprocessing, and inference pipelines live in `ml-core/aether_ml`. The FastAPI app only calls into this package.
- **Backend**: Async FastAPI app, structured into `api` (routes), `schemas` (Pydantic models), `services` (business logic), and `core` (config, logging, app wiring).
- **Frontend**: React + TypeScript + Vite skeleton for a dashboard that talks to the FastAPI backend.

### Local Development (Python parts)

From the repo root:

```bash
cd ml-core
pip install -e .

cd ../backend
pip install -e .
```

Run the API locally:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Training (Windows, `C:\mlenv`)

Assuming you have a Python 3.10 virtual environment at `C:\mlenv` and your datasets
are configured via environment variables or `config.yaml`:

```powershell
C:\mlenv\Scripts\activate
pip install -r requirements.txt
python train_change_detection.py
```

Configuration options:

- **Satellite change detection (Siamese U-Net)**
  - Environment variables:
    - `SATELLITE_CHANGE_ROOT`
    - `SATELLITE_CHANGE_TRAIN_LIST`
    - `SATELLITE_CHANGE_VAL_LIST`
  - Or `config.yaml`:
    - `satellite_change.root`
    - `satellite_change.train_list`
    - `satellite_change.val_list`

- **Aircraft classification (ViT, FGVC Aircraft)**
  - Environment:
    - `FGVC_AIRCRAFT_ROOT`
  - Or `config.yaml`:
    - `aircraft_fgvc.root`

For exporting a trained YOLOv8 aircraft detector to ONNX:

```powershell
C:\mlenv\Scripts\activate
pip install -r requirements.txt
set YOLO_WEIGHTS_PATH=path\to\yolov8_aircraft.pt
set YOLO_ONNX_OUTPUT_DIR=artifacts\onnx
python export_aircraft_detector_onnx.py
```


### Docker / Compose

Build and run the stack:

```bash
docker compose build
docker compose up -d
```

This will:

- Build the backend image, install the `aether-ml-core` package from `ml-core`, and serve FastAPI with Gunicorn/Uvicorn.
- Build the frontend image, bundle the React app, and serve static assets via Nginx.

