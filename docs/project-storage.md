# 프로젝트와 모델 저장 위치

2026-10-03에 모델 가중치와 프로젝트 실행 자료를 분리했습니다.
가중치 재학습·변환·재다운로드 없이 파일 위치와 실행 경로만 바꿨습니다.

| 위치 | 내용 |
| --- | --- |
| `Desktop/모델` | 운영 체크포인트와 PT 백업 |
| `Desktop/모델/experts/market` | 시장 전문가 원본 가중치와 가중치 압축파일 |
| `Desktop/모델/experts/stock` | 주식 정책 원본 가중치와 검증용 정책 묶음 |
| `Desktop/모델/experts/vendor` | MacroHFT 6개와 vendor 소스에 포함됐던 원본 모델 가중치 |
| `Desktop/모델/experts/fusion` | 진단용 fusion 가중치 |
| `Desktop/모델/experts/archives` | 원본 가중치가 포함된 다운로드 압축파일 |
| `artifacts/experts/checkpoints` | 전문가 config·라이선스·다운로드 메타데이터 |
| `artifacts/experts/sources` | 원본 모델 구현과 전처리 코드 |
| `artifacts/experts/native_data` | 원본 시장 입력 자료 |
| `artifacts/experts/stock-policies` | 주식 정책 소스·scaler·config·입력·측정 결과 |
| `artifacts/experts/venv`, `venv-toto` | 전문가 실행용 Python 환경 |
| `artifacts/experts/verification`, `inference` | 측정 결과·원본 출력 |
| `runtime/trading_moe` | 전문가 registry와 TradingMoE 실행·계좌 상태 |
| `runtime/markets` | 기존 Champion/Candidate 계좌·feed·replay |
| `runtime/gpu-owner.lock` | TradingMoE 실행과 전문가 진단이 공유하는 GPU 잠금 |

바탕화면 모델 폴더 최상단에는 다음 5개 운영 PT와 원본 가중치를 모은
`experts` 폴더가 있습니다. 원본의 `.pth`·`.safetensors`·정책 `.zip`·
MacroHFT `.pkl` 형식을 유지합니다.

- `champion.pt`
- `candidate.pt`
- `old-champion.pt`
- `TradingMoE.pt`
- `TradingMoE.before-stock-policies.pt`

경로의 기준은 `src/stockrl/paths.py`입니다. 시작 버튼은 프로젝트의
`artifacts/experts/venv/Scripts/python.exe`로 worker를 실행하고,
바탕화면의 해당 PT를 읽습니다. 계좌·optimizer·replay는 기존 위치에서
이어 사용하며, 화면 조회로 모델을 다시 적재하지 않습니다.

모델 폴더의 가중치와 휴지통은 Git에 포함하지 않습니다. 프로젝트의 expert 소스·config·측정 자료와 현재 가상매매에 필요한 공식 ETHUSDT 입력은 포함합니다. 설치된 Python 환경, runtime의 계좌·replay·DB·로그·캐시, 컴퓨터별 설정, vendor 예제 데이터·그림과 불필요한 대형 입력은 로컬에서 보존하고 Git에 올리지 않습니다.
운영에는 상위 PT를 사용하며, 독립 전문가 실행은 프로젝트에서 config를
읽고 모델 폴더에서 원본 가중치를 읽습니다. PT에 내장된 원본 코드의
임시 압축 해제는 기존 방식입니다.

프로젝트 루트에서 유한 구간을 직접 실행하려면:

```powershell
$expertRoot = Join-Path (Get-Location) 'artifacts/experts'
& "$expertRoot/venv/Scripts/python.exe" scripts/run_native_vertical_trading.py `
  --root $expertRoot --state runtime/trading_moe/native_vertical_run --resume --steps 3
```

이 명령은 가상매매와 학습을 실행합니다. 단순 경로 확인용으로 실행하지
마십시오. 기존 대시보드 시작/정지 버튼도 같은 경로를 사용합니다.
