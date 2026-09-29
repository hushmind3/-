# StockRL Mac 이전 폴더

이 폴더는 Windows RTX 3070 환경에서 사용하던 StockRL 프로젝트를 Apple Silicon Mac에서 이어서 실행하기 위한 사본입니다.

## Mac에서 한 번 실행

```bash
cd "StockRL-Mac-Transfer"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac.txt
chmod +x scripts/start_mac.sh scripts/start_mac.command
./scripts/start_mac.sh
```

브라우저에서 `http://127.0.0.1:8766/`를 엽니다. runtime 파일은 이 프로젝트 폴더의 `runtime-global-korea-live` 아래에 저장합니다. M5 Mac에서는 Apple Silicon용 MPS 지원 PyTorch를 설치하면 `--device auto`가 MPS를 선택하고, MPS를 사용할 수 없으면 CPU로 내려갑니다.

## 포함된 상태

- 모델 폴더의 `champion.pt`, `candidate.pt`: checkpoint 두 개
- `runtime-global-korea-live/live/agent/replay.sqlite3`: 제한 용량 replay 경험
- `runtime-global-korea-live/live/agent/paper_account.json`: 가상 계좌 상태
- `runtime-global-korea-live/live/agent/candidate_validation*.json`: 검증 상태
- `runtime-global-korea-live/live/agent/live_cursor.json`: 마지막 처리 시각
- `runtime-global-korea-live/live/market.csv`: 저장된 시장 데이터
- `configs/live_symbols_korea.json`: 한국·미국·글로벌·선물·금리·환율·코인 설정

runtime 로그와 실행 상태는 프로젝트 폴더에 둡니다. OS 자격 증명 보관함의 키를 runtime 파일로 복사하지 않습니다. 이 비공개 프로젝트 폴더를 외부에 공유하지 마세요.

## Mac에서의 성능

Mac에서는 CUDA를 사용하지 않습니다. M5 Apple Silicon에서는 MPS를 사용하며 CPU와 GPU가 통합 메모리를 공유합니다. 화면에는 별도 VRAM 대신 `MPS 통합 메모리`로 표시됩니다. 0.5B 모델의 candidate 학습은 RTX 3070보다 오래 걸릴 수 있으므로 처음에는 관찰/가상매매를 켜고 추론 상태를 확인한 뒤 자동 학습을 계속 실행합니다. checkpoint와 replay는 Windows 사본과 별개로 Mac 폴더에서 이어서 저장됩니다.
