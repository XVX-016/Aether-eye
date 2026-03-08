Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Works when invoked from backend folder:
# powershell -ExecutionPolicy Bypass -File .\scripts\start-backend.ps1

$backendRoot = Split-Path -Path $PSScriptRoot -Parent
$repoRoot = Split-Path -Path $backendRoot -Parent

Set-Location $backendRoot

$pythonCandidates = @(
    "C:\mlenv\Scripts\python.exe",
    "C:\mlenv\venv\Scripts\python.exe"
)
$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw "Python executable not found. Expected one of: $($pythonCandidates -join ', ')"
}

$env:PYTHONPATH = "$repoRoot\ml_core"
$env:BACKEND_PYTHON_EXECUTABLE = $pythonExe

& $pythonExe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
