# Continual Trading Agent — Mac 실행 안내

이 문서는 프로젝트를 Apple Silicon Mac에서 실행하는 방법입니다. 현재 설정·저장 규칙은 저장소 루트의 `AGENTS.md`와 `README.md`를 따릅니다. 저장소는 공개이므로 비밀 키·계좌 정보·운영 runtime을 커밋하지 마세요.

## 설치 및 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac.txt
./scripts/start_mac.sh
```

대시보드는 `http://127.0.0.1:8766/`에서 엽니다. `--device auto`는 지원되는 장치를 선택하며, Apple Silicon에서는 MPS를 우선 사용하고 사용할 수 없으면 CPU로 동작합니다. MPS 성능과 안정성은 해당 Mac에서 확인해야 합니다.

## 저장 위치와 운용

- runtime은 복사된 프로젝트 안 `runtime/markets/<market>/live`에 둡니다.
- 모델 가중치는 해당 컴퓨터의 바탕화면 `모델` 폴더에 `champion.pt`, `candidate.pt` 두 개만 지속 보관합니다.
- OS 자격 증명 보관함에 키를 저장합니다. 자격 증명이나 runtime 파일을 Git에 추가하지 않습니다.
- Candidate의 학습·검증·Champion 승급은 루트 README의 운영 기준을 따릅니다. 승급 검증은 같은 미래 bar와 조건에서 paper-account 순손익을 비교합니다.
- Mac 사본은 Windows 사본과 runtime·모델 파일을 공유하지 않습니다. 각 컴퓨터의 runtime과 가중치는 별도로 관리합니다.
- 실주문은 기본 OFF이며 명시적으로 허용하기 전까지 실행하지 않습니다.