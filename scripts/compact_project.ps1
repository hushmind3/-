# Move expendable vendor assets and old logs to an external, recoverable folder.
# Default is preview. Runtime/model data and Python environments stay in place.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$trashRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path $projectRoot -Parent) ((Split-Path $projectRoot -Leaf) + '-휴지통'))).TrimEnd('\')
if ($trashRoot.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw '휴지통은 프로젝트 밖에 있어야 합니다.'
}
$batchRoot = Join-Path $trashRoot (Get-Date -Format 'yyyyMMdd-HHmmss')
$moves = [Collections.Generic.List[object]]::new()
function Add-Move([string]$path, [string]$reason, [bool]$keepTail = $false) {
    $source = [IO.Path]::GetFullPath($path)
    if (-not $source.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "잘못된 원본 경로: $source" }
    $relative = $source.Substring($projectRoot.Length + 1)
    $destination = [IO.Path]::GetFullPath((Join-Path $batchRoot $relative))
    if (-not $destination.StartsWith($batchRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "잘못된 이동 경로: $destination" }
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        $moves.Add([pscustomobject]@{source=$source;destination=$destination;bytes=(Get-Item -LiteralPath $source).Length;reason=$reason;keepTail=$keepTail})
    }
}
$sources = Join-Path $projectRoot 'artifacts\experts\sources'
$demoExtensions = @('.csv','.npz','.pkl','.h5','.parquet','.feather','.png','.jpg','.jpeg','.pdf','.ipynb','.mp4','.gif')
Get-ChildItem -LiteralPath $sources -File -Recurse -Force | ForEach-Object {
    $vendor = ($_.FullName.Substring($sources.Length + 1) -split '\\')[0]
    # MacroHFT feature lists and TimesFM's live native rates input are required.
    if ($vendor -notin @('MacroHFT','TSFM_Finance') -and $_.Extension.ToLowerInvariant() -in $demoExtensions) {
        Add-Move $_.FullName '미사용 vendor 예제 데이터·그림·노트북 (실행 코드/config 유지)'
    }
}
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'artifacts\experts\stock-policies') -File -Recurse -Filter '*.ipynb' |
    ForEach-Object { Add-Move $_.FullName '미사용 주식 policy 예제 노트북 (실제 입력/코드 유지)' }
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'runtime') -File -Recurse -Filter 'cycles.jsonl' |
    Where-Object Length -GT 32MB | ForEach-Object { Add-Move $_.FullName '오래된 추론 출력 (최근 16MiB는 운영 경로 유지)' $true }
$lfs = Join-Path $projectRoot '.git\lfs\objects'
if (Test-Path -LiteralPath $lfs) {
    Get-ChildItem -LiteralPath $lfs -File -Recurse -Force | ForEach-Object { Add-Move $_.FullName 'Git LFS 로컬 다운로드 캐시 (실제 가중치/데이터는 별도 경로 유지)' }
}
$beforeBytes = (Get-ChildItem -LiteralPath $projectRoot -File -Recurse -Force | Measure-Object Length -Sum).Sum
if (-not $Apply) {
    [pscustomobject]@{preview=$true;projectGiB=[math]::Round($beforeBytes/1GB,3);files=$moves.Count;moveGiB=[math]::Round(($moves|Measure-Object bytes -Sum).Sum/1GB,3);trash=$batchRoot} | ConvertTo-Json
    exit
}
$workers = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($projectRoot) -and $_.CommandLine -match 'run_native_vertical_trading\.py|run_assembly_trial\.py|run_trading_moe_paper\.py'
}
if ($workers) { throw '모델 worker 실행 중입니다. 정지한 뒤 로그를 이동해야 합니다.' }
try { $assembly = Invoke-RestMethod 'http://127.0.0.1:8766/api/assembly/status' -TimeoutSec 2 } catch { $assembly = $null }
if ($assembly -and ($assembly.enabled -or $assembly.worker.alive)) { throw '자동조립/시험을 정지한 뒤 이동해야 합니다.' }
New-Item -ItemType Directory -Path $batchRoot -Force | Out-Null
$moves | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $batchRoot '이동목록.json') -Encoding UTF8
foreach ($item in $moves) {
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($item.destination)) -Force | Out-Null
    $tail = $null
    if ($item.keepTail) {
        $stream = [IO.File]::Open($item.source, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try {
            $start = [math]::Max(0,$stream.Length - 16MB)
            $null = $stream.Seek($start,[IO.SeekOrigin]::Begin)
            $buffer = [byte[]]::new([int]($stream.Length-$start))
            $offset=0
            while ($offset -lt $buffer.Length) {
                $count=$stream.Read($buffer,$offset,$buffer.Length-$offset)
                if ($count -eq 0) { break }; $offset+=$count
            }
            $begin=0
            if ($start -gt 0) { while ($begin -lt $offset -and $buffer[$begin] -ne 10) { $begin++ }; $begin++ }
            $tail=[byte[]]::new([math]::Max(0,$offset-$begin))
            if ($tail.Length) { [Array]::Copy($buffer,$begin,$tail,0,$tail.Length) }
        } finally { $stream.Dispose() }
    }
    Move-Item -LiteralPath $item.source -Destination $item.destination
    if ($null -ne $tail) { [IO.File]::WriteAllBytes($item.source,$tail) }
}
$afterBytes = (Get-ChildItem -LiteralPath $projectRoot -File -Recurse -Force | Measure-Object Length -Sum).Sum
$report=[pscustomobject]@{files=$moves.Count;beforeGiB=[math]::Round($beforeBytes/1GB,3);afterGiB=[math]::Round($afterBytes/1GB,3);reducedGiB=[math]::Round(($beforeBytes-$afterBytes)/1GB,3);trash=$batchRoot;deletedFiles=0}
$report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchRoot '정리결과.json') -Encoding UTF8
$report | ConvertTo-Json
