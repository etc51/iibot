param(
    [string]$Config = "configs/server_tbank_stocks_intraday_300k_focused.toml",
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8791
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $Root ".venv/Scripts/python.exe"
$LogDir = Join-Path $Root "runs/runtime"
$LogPath = Join-Path $LogDir "minimal-dashboard.log"
$PidPath = Join-Path $LogDir "minimal-dashboard.pid"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$PID | Set-Content -Path $PidPath -Encoding UTF8

& $Python -m samosbor.minimal_dashboard --config $Config --host $HostName --port $Port 2>&1 |
    ForEach-Object { Add-Content -Path $LogPath -Value $_ -Encoding UTF8 }
