$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Environment = Join-Path $ProjectRoot ".analysis_venv"
$Python = Join-Path $Environment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    py -3.11 -m venv $Environment
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-analysis.txt")
}

& $Python (Join-Path $ProjectRoot "scripts\06_analyze_public_saccades.py") @args
