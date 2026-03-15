#!/bin/bash
set -e

echo "Checking Aether-Eye services..."

BACKEND=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$BACKEND" = "200" ]; then
  echo "? Backend: OK"
else
  echo "? Backend: $BACKEND"
  exit 1
fi

FRONTEND=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$FRONTEND" = "200" ]; then
  echo "? Frontend: OK"
else
  echo "? Frontend: $FRONTEND"
  exit 1
fi

MODELS=$(curl -s http://localhost:8000/health/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','error'))")
if [ "$MODELS" = "ok" ]; then
  echo "? Models: OK"
else
  echo "? Models: $MODELS"
fi

SITES=$(curl -s http://localhost:8000/api/sites/geojson | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('features',[])))")
echo "? Sites loaded: $SITES"

echo "All checks passed."
