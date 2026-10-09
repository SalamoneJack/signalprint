# Waits for the first overnight driver to exit, then runs train.py again (completed runs are skipped,
# so this trains only experiments not yet finished, e.g. E_cnn1d_L4096) and rebuilds the report.
param([int]$WaitPid)
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if ($WaitPid) { try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch {} }
for ($i = 1; $i -le 3; $i++) {
    "[$(Get-Date -Format s)] phase2 train.py attempt $i" | Out-File -Append -Encoding utf8 logs\overnight.log
    & $py -u train.py *>> logs\train_stdout.log
    $done = (Get-ChildItem runs\*\result.json -ErrorAction SilentlyContinue).Count
    "[$(Get-Date -Format s)] phase2 attempt $i exit $LASTEXITCODE, $done result files" | Out-File -Append -Encoding utf8 logs\overnight.log
    if ($done -ge 5) { break }
}
& $py -u make_report.py *>> logs\overnight.log
"[$(Get-Date -Format s)] PHASE2 DONE" | Out-File -Append -Encoding utf8 logs\overnight.log
