param(
    [string]$Device = "auto"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "$repoRoot\src;$env:PYTHONPATH"
$env:STOCKRL_MARKET = 'korea'
$runtime = Join-Path $repoRoot 'runtime\markets\korea'
$env:STOCKRL_RUNTIME_DIR = $runtime
$modelDir = Join-Path ([Environment]::GetFolderPath('Desktop')) '모델'

Write-Host "Starting StockRL dashboard at http://127.0.0.1:8766/"

python -m stockrl web `
    --host 127.0.0.1 `
    --port 8766 `
    --runtime $runtime `
    --model-dir $modelDir `
    --device $Device `
    --config configs/live_symbols_korea.json `
    --initial-champion "$modelDir/champion.pt" `
    --candidate-every 256 `
    --fee 0.001 `
    --horizon 1m
