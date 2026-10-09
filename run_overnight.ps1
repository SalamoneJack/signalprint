# Overnight driver: download (resumable) -> train all experiments (resumable, up to 3 attempts) -> report.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force logs | Out-Null

& $py -u download_data.py *>> logs\download.log   # retries anything still missing; 403s are logged and skipped

for ($i = 1; $i -le 3; $i++) {
    "[$(Get-Date -Format s)] train.py attempt $i" | Out-File -Append -Encoding utf8 logs\overnight.log
    & $py -u train.py *>> logs\train_stdout.log
    $done = (Get-ChildItem runs\*\result.json -ErrorAction SilentlyContinue).Count
    "[$(Get-Date -Format s)] attempt $i exit $LASTEXITCODE, $done result files" | Out-File -Append -Encoding utf8 logs\overnight.log
    if ($done -ge 4) { break }
}
& $py -u make_report.py *>> logs\overnight.log
"[$(Get-Date -Format s)] OVERNIGHT DONE" | Out-File -Append -Encoding utf8 logs\overnight.log
