param(
    [string]$Python = "python",
    [string]$Data = "data/global_live.csv",
    [string]$State = "runtime-global-live",
    [string]$Config = "configs/live_symbols.json",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
foreach ($path in @($Data, $State, "logs")) {
    $parent = Split-Path -Parent $path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}

function Quote-ProcessArgument([string]$Value) {
    '"' + ($Value -replace '"', '\"') + '"'
}
function Start-Worker([string]$Name, [string[]]$Arguments) {
    $stdout = Join-Path $root "logs/$Name.out.log"
    $stderr = Join-Path $root "logs/$Name.err.log"
    $joined = ($Arguments | ForEach-Object { Quote-ProcessArgument ([string]$_) }) -join ' '
    Start-Process -FilePath $Python -ArgumentList $joined -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}

$dataPath = [IO.Path]::GetFullPath($Data)
$statePath = [IO.Path]::GetFullPath($State)
$configPath = [IO.Path]::GetFullPath($Config)
$feedStopPath = Join-Path (Split-Path -Parent $dataPath) "feed.stop"
$agentStopPath = Join-Path $statePath "stop.request"
foreach ($stopFile in @($feedStopPath, $agentStopPath)) {
    if (Test-Path -LiteralPath $stopFile) { Remove-Item -LiteralPath $stopFile -Force }
}
$feedArgs = @("-m", "stockrl", "live-feed", "--config", $configPath, "--output", $dataPath,
    "--poll-seconds", "5", "--stop-file", $feedStopPath)
$agentArgs = @("-m", "stockrl", "global-online", "--data", $dataPath, "--state-dir", $statePath,
    "--follow", "--poll-seconds", "2", "--initial-lookback-bars", "128", "--candidate-every", "256",
    "--candidate-every", "256", "--device", $Device)
$feed = $null
$agent = $null
Write-Host "Global paper agent is running. Data/replay/checkpoints persist under $dataPath and $statePath. Press Ctrl+C to stop."
try {
    $feed = Start-Worker "global-live-feed" $feedArgs
    Start-Sleep -Seconds 2
    $agent = Start-Worker "global-paper-agent" $agentArgs
    while ($true) {
        Start-Sleep -Seconds 3
        if ($feed.HasExited) {
            Write-Warning "Market collector exited ($($feed.ExitCode)); restarting. See logs/global-live-feed.err.log"
            $feed = Start-Worker "global-live-feed" $feedArgs
        }
        if ($agent.HasExited) {
            Write-Warning "Paper agent exited ($($agent.ExitCode)); restarting from its saved cursor/replay/champion. See logs/global-paper-agent.err.log"
            $agent = Start-Worker "global-paper-agent" $agentArgs
        }
    }
}
finally {
    New-Item -ItemType Directory -Force -Path $statePath | Out-Null
    New-Item -ItemType File -Force -Path $agentStopPath | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $feedStopPath) | Out-Null
    New-Item -ItemType File -Force -Path $feedStopPath | Out-Null
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline -and (($agent -and -not $agent.HasExited) -or ($feed -and -not $feed.HasExited))) {
        Start-Sleep -Milliseconds 250
        if ($agent) { $agent.Refresh() }; if ($feed) { $feed.Refresh() }
    }
    foreach ($child in @($agent, $feed)) {
        if ($null -ne $child -and -not $child.HasExited) {
            Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
