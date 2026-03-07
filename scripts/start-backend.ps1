Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location "$repoRoot\backend"

$pythonCandidates = @(
    "C:\mlenv\Scripts\python.exe",
    "C:\mlenv\venv\Scripts\python.exe"
)
$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pythonExe) {
    throw "Python executable not found. Expected one of: $($pythonCandidates -join ', ')"
}

$env:PYTHONPATH = "$repoRoot\ml-core"
$env:BACKEND_PYTHON_EXECUTABLE = $pythonExe

& $pythonExe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
