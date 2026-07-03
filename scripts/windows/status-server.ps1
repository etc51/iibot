param(
    [int]$Port = 8791
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RuntimeDir = Join-Path $Root "runs/runtime"

$paperPid = Get-Content (Join-Path $RuntimeDir "paper-loop.pid") -ErrorAction SilentlyContinue
$dashboardPid = Get-Content (Join-Path $RuntimeDir "minimal-dashboard.pid") -ErrorAction SilentlyContinue
$listener = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" } |
    Select-Object -First 1

$health = "down"
try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        $health = "ok"
    }
}
catch {
    $health = "down"
}

[PSCustomObject]@{
    PaperPid = $paperPid
    PaperRunning = [bool](Get-Process -Id $paperPid -ErrorAction SilentlyContinue)
    DashboardPid = $dashboardPid
    DashboardRunning = [bool](Get-Process -Id $dashboardPid -ErrorAction SilentlyContinue)
    Port = $Port
    Listening = [bool]$listener
    ListenerPid = if ($listener) { $listener.OwningProcess } else { $null }
    Health = $health
}
