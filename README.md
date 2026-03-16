# AETHER-EYE
### Satellite Intelligence Platform

Automated satellite imagery analysis and open-source intelligence correlation for critical infrastructure monitoring.

---

## Overview

Aether-Eye is a self-hosted intelligence platform that continuously monitors critical global sites using Sentinel-2 satellite imagery, machine learning change detection, and correlated open-source intelligence feeds. It provides operators with an integrated operations dashboard showing activity anomalies, site status, and relevant intelligence signals across all monitored locations.

The system operates entirely on-premises. No data leaves the deployment environment. All satellite imagery is sourced from the ESA Copernicus program - a free, open, government-operated data source with global coverage and a 5-day revisit cycle.

---

## Capabilities

| Capability | Detail |
|---|---|
| Satellite ingestion | Automated Sentinel-2 L2A scene discovery via Copernicus STAC |
| Change detection | SiameseUNet ML model, test IoU 0.82 on building change dataset |
| Activity baselines | Per-site temporal baselines with anomaly scoring |
| Intelligence feed | 10 RSS sources geo-tagged to monitored sites |
| ADS-B integration | Live aircraft positions via OpenSky Network |
| Operations dashboard | Global map, site detail, event feed, intel correlation |
| Alert levels | NORMAL / ELEVATED (1.5x baseline) / ANOMALOUS (2x baseline) |

---

## Monitored Sites

18 sites across four categories:

**Military Airbases**
| Site | Country | Priority |
|---|---|---|
| Al Dhafra Air Base | UAE | Critical |
| Al Udeid Air Base | Qatar | Critical |
| Diego Garcia | BIOT | Critical |
| Kadena Air Base | Japan | Critical |
| Andersen AFB | Guam | Critical |
| Ramstein Air Base | Germany | High |
| Incirlik Air Base | Turkey | High |
| Al-Asad Air Base | Iraq | High |
| Bagram Air Base | Afghanistan | Medium |

**Naval Bases**
| Site | Country | Priority |
|---|---|---|
| Naval Station Norfolk | USA | High |
| Naval Station Rota | Spain | High |
| Pearl Harbor Naval Base | USA | High |
| Changi Naval Base | Singapore | High |

**Strategic Ports**
| Site | Country | Priority |
|---|---|---|
| Bandar Abbas Port | Iran | Critical |
| Port of Aden | Yemen | High |
| Jeddah Islamic Port | Saudi Arabia | Medium |

**Civil Airports**
| Site | Country | Priority |
|---|---|---|
| Dubai International Airport | UAE | High |
| Abu Dhabi International | UAE | Medium |

---

## Architecture

```text
Sentinel-2 STAC (Copernicus)
         |
         v
   STAC Watcher --> Scene Processor --> Change Detection Model
   (APScheduler)         |                  (SiameseUNet ONNX)
                         |
                         v
                   Tile Detections
                         |
                         v
              Event Engine --> Activity Alerts
              (temporal         (NEW_OBJECT /
               baselines)        ELEVATED /
                                 ACTIVITY_SURGE)
                         |
                         v
                     PostGIS DB
                         |
              +----------+----------+
              v                     v
         FastAPI                Intel Feed
         REST API            (RSS ingestion,
              |                geo-tagging)
              v
      Next.js Dashboard
      (Operations Map,
       Site Detail,
       Intel Correlation)
```

---

## Quick Start

Prerequisites: Docker, Docker Compose

```text
git clone https://github.com/XVX-016/Aether-eye.git
cd Aether-eye
cp .env.example .env
docker compose up -d
```

Dashboard:  http://localhost:3000
API docs:   http://localhost:8000/docs

Demo mode (populates dashboard with realistic data):

```text
# Windows
.\scripts\demo_start.ps1
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI, Python 3.10 |
| ML inference | PyTorch, ONNX Runtime |
| Change detection | SiameseUNet (custom trained) |
| Database | PostgreSQL 16 + PostGIS 3.4 |
| ORM | SQLAlchemy 2.0 async |
| Frontend | Next.js 14, TypeScript |
| Map | MapLibre GL |
| Satellite data | Sentinel-2 L2A via Copernicus STAC |
| Scheduling | APScheduler |
| Containerization | Docker, Docker Compose |

---

## Data Sources

| Source | Type | License |
|---|---|---|
| ESA Copernicus Sentinel-2 | Satellite imagery | Free, open government data |
| OpenSky Network | ADS-B aircraft positions | Free for non-commercial use |
| BBC News, Sky News | RSS intelligence | Public |
| Breaking Defense, The Aviationist | RSS intelligence | Public |
| Al Jazeera, Arab News, The National | RSS intelligence | Public |
| Naval News, Middle East Eye | RSS intelligence | Public |

---

## Model Performance

| Model | Architecture | Dataset | Val IoU | Test IoU |
|---|---|---|---|---|
| Change Detection | SiameseUNet | Building-change (1,134 pairs) | 0.7936 | 0.8243 |

---

## Deployment

The system is fully self-hosted and supports air-gapped deployment with pre-downloaded satellite imagery. No cloud dependencies are required for core detection and monitoring functionality.

Minimum recommended specification:
- 4 CPU cores
- 8 GB RAM (16 GB recommended for ML inference)
- 50 GB storage
- Ubuntu 22.04 LTS or Windows Server 2022

---

## Project Structure

```text
backend/
  app/        FastAPI application, routes, schemas, database models
  pipeline/   Scene tiling, change detection, event engine, STAC watcher
  services/   Intelligence feed ingestion, ADS-B integration
  configs/    Global site registry, STAC config, inference config
  alembic/    Database migrations

frontend/
  app/        Next.js App Router pages
  components/ Dashboard components (map, feed, site detail, timeline)

ml_core/
  aether_ml/  Model training, evaluation, dataset loaders
  artifacts/  Trained model checkpoints (change_model_v2)

scripts/
  seed_demo_data.py   Populate dashboard with demo data
  demo_start.ps1      One-command demo launcher (Windows)
  healthcheck.sh      Service validation script

docker-compose.yml    Full stack deployment
```

---

## License

Proprietary. All rights reserved.
