# Production Deployment Guide

This document is a comprehensive, step-by-step deployment manual for the Aether-Eye platform, covering local development, fully containerized Docker deployments, and cloud production environments.

---

## 1. Prerequisites & Specifications

### Software Requirements
*   **Operating Systems**: Windows 10/11, Ubuntu 20.04+, macOS 12+
*   **Docker Engine**: v24.0.0 or higher
*   **Docker Compose**: v2.20.0 or higher
*   **Python Runtime**: v3.10.x (v3.10 is strictly recommended for pipeline packages)
*   **Node.js Runtime**: v20.x (LTS)
*   **Git**: v2.40+ with **Git LFS (Large File Storage)** installed
*   **PostgreSQL/PostGIS**: PostgreSQL 15/16 with PostGIS 3.4+ (if deploying databases outside Docker)

### Hardware Requirements
| Resource | Minimum Specifications | Recommended Specifications |
| :--- | :--- | :--- |
| **Processor** | 4 CPU Cores (x86_64 or ARM64) | 8+ CPU Cores |
| **System RAM** | 8 GB RAM | 16+ GB RAM |
| **Disk Space** | 20 GB free space | 50+ GB SSD (to cache satellite GeoTIFF tiles) |

### GPU Acceleration Requirements (Optional)
By default, Aether-Eye processes ONNX and PyTorch model inferences using optimized multi-threaded CPU execution. To leverage hardware acceleration:
*   **Supported GPUs**: NVIDIA GPU (Turing, Ampere, Ada Lovelace, or Hopper architectures).
*   **NVIDIA Drivers**: Version 525+ or higher.
*   **Toolkit Integration**: CUDA Toolkit 11.8 or 12.x alongside matching cuDNN 8.9+ libraries.
*   **Python Driver**: Install the GPU execution provider using:
    ```bash
    pip install onnxruntime-gpu
    ```

---

## 2. Environment 1: Local Development Setup

To configure Aether-Eye directly on a local development workstation, follow these steps:

### Step 2.1: Clone the Repository with Large File Storage (LFS)
Deep learning weights are tracked using Git LFS. You must initialize Git LFS before cloning:
```bash
# Initialize LFS on your workstation
git lfs install

# Clone the repository
git clone <repository_url> Aether-eye
cd Aether-eye

# Pull and verify the binary weights
git lfs pull
```

### Step 2.2: Set Up the Spatial Database (Docker)
The easiest way to run PostgreSQL with PostGIS is using the official PostGIS Docker image:
```bash
docker run --name aether-db \
  -e POSTGRES_DB=aether_eye \
  -e POSTGRES_USER=aether \
  -e POSTGRES_PASSWORD=aether \
  -p 5432:5432 \
  -d postgis/postgis:16-3.4
```

### Step 2.3: Configure the Python Backend
1.  **Create and Activate Virtual Environment**:
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install Ingestion & Inference Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    pip install -e ml_core
    ```
3.  **Configure Local Environment Variables**:
    Copy the example configuration to a local active `.env` file:
    ```bash
    cp .env.example .env
    ```
    Ensure that `DATABASE_URL` in `.env` is pointed at your local database:
    ```ini
    DATABASE_URL=postgresql+asyncpg://aether:aether@localhost:5432/aether_eye
    ```
4.  **Execute Database Migrations**:
    Apply the database tables using Alembic:
    ```bash
    alembic --config backend/alembic.ini upgrade head
    ```
5.  **Launch the FastAPI Server**:
    Start the backend with reload capabilities enabled:
    ```bash
    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
    ```

### Step 2.4: Configure the Next.js Frontend
1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```
2.  **Install Node.js dependencies**:
    ```bash
    npm install
    ```
3.  **Start the Local Frontend Dev Server**:
    ```bash
    npm run dev
    ```
    The frontend dev server starts at `http://localhost:3000`. Same-origin API calls are proxied directly to the backend at `http://localhost:8000/api` using the catch-all proxy route.

### Step 2.5: Verify Local Diagnostics
1.  Verify the backend health check endpoint:
    ```bash
    curl http://localhost:8000/health
    # Expected: {"status":"ok","version":"1.1.0"}
    ```
2.  Verify the model load state:
    ```bash
    curl http://localhost:8000/health/models
    # Expected: {"status":"ok","change_model":"v2","aircraft_classifier":"convnext_small_100cls"}
    ```

### Step 2.6: Load Demo Data
Run the following script to seed the local database with mock temporal scenes, flights, intel articles, and detections:
```bash
python scripts/seed_demo_data.py
```

---

## 3. Environment 2: Full Docker Containerized Deployment

This is the recommended environment for standard on-premises or server deployments. It packages all components into three isolated containers under a unified network.

### Step 3.1: Configure the Production `.env`
Copy `.env.example` to `.env` at the repository root and verify that `DATABASE_URL` is set to point to the internal Docker database host `db` rather than `localhost`:
```ini
DATABASE_URL=postgresql+asyncpg://aether:aether@db:5432/aether_eye
BACKEND_API_BASE_URL=http://backend:8000/api
```

### Step 3.2: Start the Container Stack
Start the database, backend, and frontend containers in detached mode:
```bash
docker compose up -d --build
```
This automatically applies migrations on startup, mounts model weights in read-only mode, and starts the scheduled crawlers.

### Step 3.3: Verify Container Health
Run the automated system health check script:
```bash
# On Linux/macOS
bash scripts/healthcheck.sh

# On Windows
powershell ./scripts/demo_start.ps1
```
This script tests:
*   FastAPI health status (`200 OK`)
*   Next.js dashboard response (`200 OK`)
*   Deep learning models loading status (`ok`)
*   Monitored locations and spatial layers loaded (`18 monitored sites`)

### Step 3.4: Seed Production Demo Data
Run the demo seeding script inside the running backend container:
```bash
docker exec -it aether-backend python scripts/seed_demo_data.py
```
Open `http://localhost:3000` to access the operations dashboard.

---

## 4. Environment 3: Cloud Deployment (Supabase + Render + Vercel)

This environment describes how to deploy the platform using standard cloud providers: **Supabase** for database hosting, **Render** for the Python backend, and **Vercel** for the React frontend.

```
+------------------+       +------------------+       +-------------------+
|  Vercel Frontend | ----> |  Render Backend  | ----> | Supabase Database |
|  Next.js Static  |       |  FastAPI Server  |       | PostgreSQL +      |
|  Distribution    |       |  & Model Runner  |       | PostGIS Extension |
+------------------+       +------------------+       +-------------------+
```

### Step 4.1: Configure Supabase Database
1.  Create a new project in the [Supabase Dashboard](https://supabase.com/).
2.  Open the **SQL Editor** in Supabase and enable the PostGIS spatial extension:
    ```sql
    CREATE EXTENSION IF NOT EXISTS postgis;
    ```
3.  Go to **Project Settings > Database** and copy your connection string URI (use the Transaction Pooler or Session Pooler URI on port `5432` or `6543`).
4.  **Important**: The URI copied from Supabase uses the standard `postgresql://` prefix. You **must** change this to `postgresql+asyncpg://` to enable SQLAlchemy's async driver support:
    ```
    # Original Supabase Connection String:
    postgresql://postgres.xxx:PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres
    
    # Transformed Connection String for SQLAlchemy:
    postgresql+asyncpg://postgres.xxx:PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres
    ```

### Step 4.2: Deploy Backend to Render
1.  Create a new **Web Service** on [Render](https://render.com/) and connect it to your GitHub repository.
2.  Set the following build parameters:
    *   **Runtime**: `Python 3.10`
    *   **Build Command**: `pip install -r backend/requirements.txt && pip install ./ml_core`
    *   **Start Command**: `alembic --config backend/alembic.ini upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1`
3.  Configure the **Instance Type**:
    > [!IMPORTANT]
    > You **must** choose Render's **Starter** instance tier ($7/month, 512 MB to 1 GB RAM) or higher.
    > The Free tier will run out of memory during ONNX runtime model initialization and crash with exit code `137` (OOM).
4.  Configure the following **Environment Variables** in the Render settings:
    *   `DATABASE_URL`: Your transformed `postgresql+asyncpg://` Supabase connection string.
    *   `PYTHONPATH`: `/opt/render/project/src`
    *   `ENABLE_STAC_WATCHER`: `false` (to avoid high resource consumption).
    *   `ENABLE_ACTIVITY_AGGREGATOR`: `true`
    *   `ENABLE_INTEL_FETCH_ON_STARTUP`: `true`
    *   `ENABLE_FLIGHT_FETCH_ON_STARTUP`: `true`

### Step 4.3: Deploy Frontend to Vercel
1.  Create a new project on [Vercel](https://vercel.com/) and connect it to your GitHub repository.
2.  Set the following build parameters:
    *   **Framework Preset**: `Next.js`
    *   **Root Directory**: `frontend`
3.  Configure the following **Environment Variables**:
    *   `BACKEND_API_BASE_URL`: The URL of your deployed Render backend (e.g., `https://aether-backend.onrender.com/api`).
4.  Deploy the project. The frontend catch-all API proxy handles same-origin routing using this base URL.

### Step 4.4: Apply Migrations & Seed Data Remotely
To apply database migrations and seed the demo data on your remote Supabase database, run the scripts from your local machine pointing to the remote database URL:
```bash
# Windows
$env:DATABASE_URL="postgresql+asyncpg://postgres.xxx:PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
alembic --config backend/alembic.ini upgrade head
python scripts/seed_demo_data.py

# Linux/macOS
export DATABASE_URL="postgresql+asyncpg://postgres.xxx:PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
alembic --config backend/alembic.ini upgrade head
python scripts/seed_demo_data.py
```

---

## 5. Managing Model Artifacts

Deep learning models must be correctly downloaded and verified for the system to boot successfully.

### Manual Download Fallback
If Git LFS fails or the Git LFS bandwidth limit is exceeded, you can download the weights manually.
1.  Create the following target directories:
    *   `ml_core/artifacts/change_model_v2/`
    *   `ml_core/artifacts/aircraft_classifier_v1/`
2.  Download the production weight files and place them in their respective directories:
    *   **Change Detection Model v2**: Place `change_model_v2.pt` and `change_model_v2.onnx` inside `ml_core/artifacts/change_model_v2/`.
    *   **Aircraft Classifier v1**: Place `aircraft_classifier_v1.pt` and `aircraft_classifier_v1.onnx` inside `ml_core/artifacts/aircraft_classifier_v1/`.

### Verifying Model Integrity
To verify that the weights have downloaded correctly and are not LFS placeholder files, run the health check endpoint:
```bash
curl http://localhost:8000/health/models
```
If the files are missing or are invalid LFS placeholders, the server returns a `503 Service Unavailable` response detailing the path resolution failure.

---

## 6. Pre-Production Launch Checklist

Before opening the dashboard to users, verify that all steps on this checklist are complete:

*   [ ] **API Responsiveness**: `GET /health` returns status `200 OK`.
*   [ ] **Model Verification**: `GET /health/models` returns status `200 OK` and lists active models as `v2` and `convnext_small_100cls`.
*   [ ] **Database Schema**: The output of `alembic --config backend/alembic.ini current` matches the latest revision in `alembic/versions/`.
*   [ ] **Feeds Check**: The database `flight_states` and `intel_articles` tables are populated.
*   [ ] **Sites Loaded**: The `GET /api/sites/geojson` endpoint returns a FeatureCollection containing **all 18 monitored locations** defined in `global_sites.yaml`.

---

## 7. Troubleshooting & Common Issues

### Issue 7.1: SSL Connection Refused on `asyncpg`
*   **Error Message**: `sqlalchemy.exc.InterfaceError: Cannot connect to host ... [SSL: CERTIFICATE_VERIFY_FAILED]` or `ssl error: sslmode value "require" invalid when SSL support is not compiled in`
*   **Cause**: Cloud database providers (like Supabase or AWS RDS) require SSL encryption, but the local connection pool has SSL disabled.
*   **Fix**: Update `backend/app/database/session.py` to enable SSL parameters for connections, or append `?ssl=require` or `?sslmode=require` to your `DATABASE_URL` connection string:
    ```ini
    DATABASE_URL=postgresql+asyncpg://postgres.xxx:PASSWORD@pooler.supabase.com:5432/postgres?ssl=require
    ```

### Issue 7.2: MapLibre Basemap Tiles Fail to Load
*   **Error Message**: `Map initialization failed: WebGL not supported` or `Primary map style blocked/failed` (Common in Firefox)
*   **Cause**: Strict security tracking blockers (such as Enhanced Tracking Protection in Firefox) block the default CartoDB vector basemaps.
*   **Fix**: Aether-Eye includes an automatic fallback mechanism. If the primary styling fails to load within 3 seconds, the map automatically falls back to rendering standard local MapLibre demo tiles. You can verify this by checking the console log for the warning:
    `Primary map style blocked/failed, falling back to MapLibre demo tiles`

### Issue 7.3: ONNX Model Not Found on Startup
*   **Error Message**: `RuntimeError: Change model checkpoint missing: ...` or `ONNX model load failed: Invalid Protobuf File`
*   **Cause**: The model checkpoints are LFS pointer files (which are only a few bytes in size) instead of the actual binary weights.
*   **Fix**: Navigate to your project directory and run `git lfs pull` to download the actual model binary weights.

### Issue 7.4: Flight Feed OpenSky Network API Returns Empty
*   **Error Message**: `Flight feed failed [al_udeid]: HTTP 429 Too Many Requests` or `OpenSky connection timed out`
*   **Cause**: The public OpenSky Network API imposes strict rate limits on anonymous queries.
*   **Fix**: Add valid OpenSky API credentials in your environment configuration, or switch to local network parsing using an ADS-B receiver (such as dump1090) in air-gapped environments.

### Issue 7.5: Windows `RuntimeError: Event loop is closed`
*   **Error Message**: `RuntimeError: Event loop is closed` (printed in console when shutting down the FastAPI server)
*   **Cause**: This is a harmless warning printed by Python's `asyncio` on Windows when closing the Proactor event loop during server shutdown.
*   **Fix**: No action is required. This warning does not affect the performance, database integrity, or reliability of the platform.
