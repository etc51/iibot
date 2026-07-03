param(
    [string]$TaskName = "MOEX AI Trader Server",
    [string]$Config = "configs/server_tbank_stocks_intraday_300k_focused.toml",
    [int]$Port = 8791,
    [int]$IntervalSeconds = 900
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$StartScript = Join-Path $Root "scripts/windows/start-server.ps1"
$Argument = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$StartScript`"",
    "-Config", "`"$Config`"",
    "-HostName", "0.0.0.0",
    "-Port", "$Port",
    "-IntervalSeconds", "$IntervalSeconds"
) -join " "

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $Argument `
    -WorkingDirectory $Root

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Starts MOEX AI Trader paper loop and dashboard." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
