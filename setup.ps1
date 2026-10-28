# One-time setup. Run with:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Continue"

# Find Python: the "py" launcher first (most reliable on Windows), then "python".
$py = $null
foreach ($candidate in @("py", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        & $candidate --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $py = $candidate; break }
    }
}
if (-not $py) {
    Write-Host "Python was not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python command: $py" -ForegroundColor Cyan
& $py --version

Write-Host "Creating the virtual environment (.venv)..." -ForegroundColor Cyan
& $py -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Host "Could not create the virtual environment." -ForegroundColor Red; exit 1 }

Write-Host "Installing libraries (this takes a minute or two)..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Installing the libraries failed. Copy the error above and send it to Claude." -ForegroundColor Red; exit 1 }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created the .env file." -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup finished!" -ForegroundColor Green
Write-Host "NEXT: open the file named .env in Notepad, replace your-key-here with your Gemini API key, and save."
Write-Host "Then run:  powershell -ExecutionPolicy Bypass -File .\run.ps1"
