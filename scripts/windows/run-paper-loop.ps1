param(
    [string]$Config = "configs/server_tbank_stocks_intraday_300k_focused.toml",
    [int]$IntervalSeconds = 900
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $Root ".venv/Scripts/python.exe"
$LogDir = Join-Path $Root "runs/runtime"
$LogPath = Join-Path $LogDir "paper-loop.log"
$PidPath = Join-Path $LogDir "paper-loop.pid"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$PID | Set-Content -Path $PidPath -Encoding UTF8

function Write-LoopLog {
    param([string]$Message)
    $Stamp = (Get-Date).ToString("o")
    Add-Content -Path $LogPath -Value "[$Stamp] $Message" -Encoding UTF8
}

Write-LoopLog "paper loop started config=$Config interval=${IntervalSeconds}s"

while ($true) {
    try {
        Write-LoopLog "paper-cycle start"
        & $Python -m samosbor.cli --config $Config paper-cycle 2>&1 |
            ForEach-Object { Add-Content -Path $LogPath -Value $_ -Encoding UTF8 }
        $ExitCode = $LASTEXITCODE
        Write-LoopLog "paper-cycle finished exit_code=$ExitCode"
    }
    catch {
        Write-LoopLog "paper-cycle exception: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $IntervalSeconds
}
