$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$AppUrl = "http://127.0.0.1:8000/"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found at $PythonExe. Create it first with: python -m venv venv"
}

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Warning "ANTHROPIC_API_KEY is not set. The app runs, but AI explanations may show an auth error."
}

Write-Host "Opening app at $AppUrl ..."
Start-Process $AppUrl

Write-Host "Starting backend server..."
Set-Location $ProjectRoot
& $PythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
