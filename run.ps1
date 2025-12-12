param(
  [ValidateSet("app", "dynamic-pe", "tests")]
  [string]$Target = "app",

  [string]$Code = "600519.SH",
  [string]$Start = "",
  [string]$End = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Write-Host "Missing .venv. Create it first:" -ForegroundColor Yellow
  Write-Host "  python -m venv .venv" -ForegroundColor Yellow
  exit 2
}

if ($Target -eq "app") {
  & $venvPy -m streamlit run main.py
  exit $LASTEXITCODE
}

if ($Target -eq "dynamic-pe") {
  if ($Start -and $End) {
    & $venvPy scripts/get_dynamic_pe.py --code $Code --start $Start --end $End
  } else {
    & $venvPy scripts/get_dynamic_pe.py --code $Code
  }
  exit $LASTEXITCODE
}

if ($Target -eq "tests") {
  & $venvPy -m pytest -q
  exit $LASTEXITCODE
}
