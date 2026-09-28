param(
    [int]$Port = 8765,
    [string]$Device = "auto"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "$repoRoot\src;$env:PYTHONPATH"

$requestedPort = $Port
while ($Port -le 8799) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        break
    } catch {
        if ($listener) { $listener.Stop() }
        $Port++
    }
}
if ($Port -gt 8799) { throw "No free dashboard port found between $requestedPort and 8799." }
if ($Port -ne $requestedPort) { Write-Host "Port $requestedPort is occupied; using $Port instead." }
Write-Host "Starting StockRL dashboard at http://127.0.0.1:$Port/"

python -m stockrl web `
    --host 127.0.0.1 `
    --port $Port `
    --runtime runtime-global-korea-live `
    --device $Device `
    --config configs/live_symbols_korea.json `
    --initial-champion runtime-global-korea-live/live/agent/champion.pt `
    --candidate-every 256 `
    --fee 0.001 `
    --horizon 1m