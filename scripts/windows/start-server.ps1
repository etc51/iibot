param(
    [string]$Config = "configs/server_tbank_stocks_intraday_300k_focused.toml",
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8791,
    [int]$IntervalSeconds = 900,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RuntimeDir = Join-Path $Root "runs/runtime"
$PaperScript = Join-Path $Root "scripts/windows/run-paper-loop.ps1"
$DashboardScript = Join-Path $Root "scripts/windows/run-min-dashboard.ps1"
$ServerPidPath = Join-Path $RuntimeDir "server.pid"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$PID | Set-Content -Path $ServerPidPath -Encoding UTF8

function Stop-ExistingServerProcesses {
    $escapedRoot = [regex]::Escape($Root)
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.CommandLine -match $escapedRoot -and
        (
            $_.CommandLine -like "*run-paper-loop.ps1*" -or
            $_.CommandLine -like "*run-min-dashboard.ps1*" -or
            $_.CommandLine -like "*samosbor.minimal_dashboard*"
        )
    }
    foreach ($target in $targets) {
        Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-DashboardFirewallRule {
    if (-not (Get-Command New-NetFirewallRule -ErrorAction SilentlyContinue)) {
        return "firewall-cmdlets-unavailable"
    }
    $name = "MOEX AI Trader Dashboard 8791"
    $existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Enable-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Out-Null
        return "firewall-rule-existing"
    }
    try {
        New-NetFirewallRule `
            -DisplayName $name `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -LocalPort $Port `
            -Profile Any | Out-Null
        return "firewall-rule-created"
    }
    catch {
        return "firewall-rule-failed: $($_.Exception.Message)"
    }
}

if (-not $KeepExisting) {
    Stop-ExistingServerProcesses
    Start-Sleep -Seconds 2
}

$firewallStatus = Ensure-DashboardFirewallRule

Start-Process `
    -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PaperScript, "-Config", $Config, "-IntervalSeconds", "$IntervalSeconds") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden

Start-Process `
    -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $DashboardScript, "-Config", $Config, "-HostName", $HostName, "-Port", "$Port") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden

Start-Sleep -Seconds 4

$paperPid = Get-Content (Join-Path $RuntimeDir "paper-loop.pid") -ErrorAction SilentlyContinue
$dashboardPid = Get-Content (Join-Path $RuntimeDir "minimal-dashboard.pid") -ErrorAction SilentlyContinue
$listener = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" } |
    Select-Object -First 1

[PSCustomObject]@{
    Root = $Root
    Config = $Config
    PaperPid = $paperPid
    DashboardPid = $dashboardPid
    Port = $Port
    Listening = [bool]$listener
    ListenerPid = if ($listener) { $listener.OwningProcess } else { $null }
    Firewall = $firewallStatus
}
