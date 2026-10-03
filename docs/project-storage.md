# 프로젝트와 모델 저장 위치

2026-10-03에 기존 다운로드 자료를 프로젝트 안으로 이동했습니다.
가중치 재학습·변환·재다운로드 없이 파일 위치와 실행 경로만 바꿨습니다.

| 위치 | 내용 |
| --- | --- |
| `Desktop/모델` | 운영 체크포인트와 PT 백업만 보관 |
| `artifacts/experts/checkpoints` | 다운로드한 전문가 원본 가중치·config |
| `artifacts/experts/sources` | 원본 모델 구현과 전처리 코드 |
| `artifacts/experts/native_data` | 원본 시장 입력 자료 |
| `artifacts/experts/stock-policies` | 주식 정책 원본 가중치·scaler·입력·측정 결과 |
| `artifacts/experts/venv`, `venv-toto` | 전문가 실행용 Python 환경 |
| `artifacts/experts/verification`, `inference`, `fusion` | 측정 결과·원본 출력·진단용 상태 |
| `runtime/trading_moe` | 전문가 registry와 TradingMoE 실행·계좌 상태 |
| `runtime/markets` | 기존 Champion/Candidate 계좌·feed·replay |
| `runtime/gpu-owner.lock` | TradingMoE 실행과 전문가 진단이 공유하는 GPU 잠금 |

바탕화면 모델 폴더에는 다음 5개 파일만 남습니다.

- `champion.pt`
- `candidate.pt`
- `old-champion.pt`
- `TradingMoE.pt`
- `TradingMoE.before-stock-policies.pt`

경로의 기준은 `src/stockrl/paths.py`입니다. 시작 버튼은 프로젝트의
`artifacts/experts/venv/Scripts/python.exe`로 worker를 실행하고,
바탕화면의 해당 PT를 읽습니다. 계좌·optimizer·replay는 기존 위치에서
이어 사용하며, 화면 조회로 모델을 다시 적재하지 않습니다.

전문가 원본 다운로드·실행 환경은 용량이 크므로 Git에서 제외합니다.
원본 가중치는 프로젝트 내부 백업 자료로 보존하고, 운영에는 상위 PT를
사용합니다. PT에 내장된 원본 코드의 임시 압축 해제는 기존 방식입니다.

프로젝트 루트에서 유한 구간을 직접 실행하려면:

```powershell
$expertRoot = Join-Path (Get-Location) 'artifacts/experts'
& "$expertRoot/venv/Scripts/python.exe" scripts/run_native_vertical_trading.py `
  --root $expertRoot --state runtime/trading_moe/native_vertical_run --resume --steps 3
```

이 명령은 가상매매와 학습을 실행합니다. 단순 경로 확인용으로 실행하지
마십시오. 기존 대시보드 시작/정지 버튼도 같은 경로를 사용합니다.
