# SCP fallback. Prefer GitHub: git push on the laptop, git pull on AutoDL.
param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$User = "root",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)

$ErrorActionPreference = "Stop"
$Archive = Join-Path $env:TEMP "D-EMAML-repro.tar.gz"

if (Test-Path $Archive) { Remove-Item $Archive -Force }

tar -czf $Archive `
    --exclude=.git `
    --exclude=.venv `
    --exclude=artifacts `
    --exclude=__pycache__ `
    --exclude=.pytest_cache `
    -C $ProjectRoot .

scp -P $Port $Archive "${User}@${HostName}:/root/autodl-tmp/D-EMAML-repro.tar.gz"
ssh -p $Port "${User}@${HostName}" `
    "mkdir -p /root/autodl-tmp/D-EMAML-repro && tar -xzf /root/autodl-tmp/D-EMAML-repro.tar.gz -C /root/autodl-tmp/D-EMAML-repro && rm /root/autodl-tmp/D-EMAML-repro.tar.gz"

Write-Host "Uploaded to /root/autodl-tmp/D-EMAML-repro"
