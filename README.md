# Continual Trading Agent · TradingMoE

여러 pretrained expert를 독립 모듈로 유지하는 TradingMoE와 가상매매·학습·자동조립 운영 프로젝트입니다. Python backend와 React + Vite + TypeScript 프론트엔드를 사용합니다.

## 설치와 실행

Windows 새 컴퓨터에서는 다음 순서로 실행합니다.

1. Python 3.13과 설치 스크립트의 CUDA PyTorch를 지원하는 NVIDIA 드라이버를 준비합니다.
2. 이 저장소를 다운로드하거나 clone하고 `설치.cmd`를 실행합니다.
3. 모델 가중치는 별도로 준비해 사용자 바탕화면의 `모델` 폴더에 둡니다.
4. `서버켜기.cmd`를 실행하고 `http://127.0.0.1:8766/`을 엽니다.
5. 화면에서 필요한 Feed와 모델을 각각 시작합니다. 서버 시작만으로 모델을 자동 적재하지 않습니다.

운영 UI의 빌드 결과물이 저장소에 포함되어 있으므로 **Python 서버만으로 전체 화면이 열립니다. Node/Vite는 운영에 필요하지 않습니다.**

설치 환경·모델 전달·빈 DB 생성은 [Windows 설치 안내](docs/windows-install.md), 저장 위치는 [프로젝트 저장 구조](docs/project-storage.md)를 참고합니다.

## 현재 모델과 실행 경로

- 상위 모델: `TradingMoE.pt`. 현재 기본 Registry에는 시장 인식 8개, MacroHFT 정책 6개, 주식 매매 정책 6개로 총 20개 expert가 있습니다. 화면은 실제 Registry를 읽으며 이름이나 개수를 고정하지 않습니다.
- 내부 expert는 서로 가중치를 평균하지 않고 독립 모듈로 보존합니다. Adapter, Router, Fusion, Controller가 시장 정보와 정책 의견을 연결합니다.
- 실행 경로: 시장 입력 → expert → market state / policy evidence → controller → 가상 주문·체결 → 계좌 NAV/P&L → reward → replay → optimizer update.
- 현재 전용 연속 worker는 확보한 **공식 ETHUSDT 과거 구간**을 실행합니다. 별도 실시간 Feed의 수집 종목 수를 MoE의 실시간 매매 종목 수로 해석하면 안 됩니다.
- MarketGPT의 실제 AAPL ITCH 기록은 원래 시점을 유지하는 과거 자료입니다. ETH와 같은 시점의 호가라고 표시하지 않습니다. 현재 경로 구동 결과는 데이터 정합성이나 수익성을 입증하지 않습니다.
- 원본 expert 가중치와 학습 가능 그룹은 구분하며, 업데이트 그룹은 실행 설정에 따릅니다. GPU expert는 동시에 최대 1개씩 실행하고, 다른 worker가 GPU 잠금을 사용하면 차례를 기다립니다.
- `TradingMoE.pt`는 worker 시작 때 한 번 적재합니다. 화면 조회·새로고침으로 다시 적재하지 않습니다. 정지는 계좌·optimizer·replay와 모델 저장 후 종료합니다.

자세한 구조는 [TradingMoE 학습 경로](docs/trading-moe-learning.md), [전용 lifecycle](docs/trading-moe-runtime.md), [주식 정책 expert](docs/stock_policy_experts.md)에 있습니다.

## 운영 화면

| 화면 / 주소 | 기능 |
| --- | --- |
| 운영 · 계좌 `/#control` | 운영 상태, 독립 판단/체결/학습 제어, Feed, Champion/Candidate 시작·저장 후 정지, 계좌·체결·손익 |
| 시장 · 판단 `/#markets` | 시장 수신, 종목 검색·필터, 입력·판단 상태 |
| 경험 학습 `/#learning` | optimizer, loss, reward, 학습량·시간, 결과 대기·학습 가능·완료, replay |
| 승급전 `/#promotionTrial` | 시험 진행, 계좌 비교, 승격 기록 |
| 연결 설정 `/#connection` | 시세 환경·인증·연결 상태 |
| 상세 · 기록 `/#system` | GPU/RAM, runtime, 오류·로그·세부 성능 |
| TradingMoE · 전문가 `/#experts` | 실제 Registry, 적재·활성·dtype·메모리, 원본 출력·shape, 추론 측정 |
| TradingMoE · 자동매매 `/#trading-moe` | 전용 worker 시작/정지, 판단·가상체결·NAV·손익·reward·loss·optimizer |
| 조립 · 자동실험 `/#assembly` | 후보 생성·시험·탈락·교체, 자동화 설정, expert 비교·queue·history |

Champion과 Candidate는 각자의 모델·계좌·실행 상태를 갖습니다. 둘 중 하나만 실행하거나 둘 다 정지한 채 Feed만 실행할 수 있습니다. 전용 TradingMoE worker는 별도 lifecycle입니다. 장기 운영계좌와 승급전 시험계좌는 구분합니다. 기존 Transformer 지원 코드는 호환 경로로 남아 있으며, 그 승격 조건을 TradingMoE 자동조립 평가 조건으로 취급하지 않습니다.

자동조립 Candidate는 공용 expert 본체와 작은 recipe/router/fusion/controller state를 사용하며 후보마다 전체 PT를 복사하지 않습니다. [자동조립 구조와 평가](docs/assembly.md)

## 프론트엔드 수정

새 UI 소스는 `.tsx` 컴포넌트와 `.ts` API·타입·hook으로 구성됩니다. 구형 `web_dashboard.html`과 `web/assets`의 HTML/JS/CSS 화면은 제거했습니다.

```powershell
cd frontend
npm ci
npm run dev
npm run typecheck
npm test
npm run build
```

개발·빌드에 Node.js LTS가 필요합니다. 개발 서버는 `http://127.0.0.1:5173/`이며 기존 Python API로 요청을 전달합니다. 빌드 결과는 `src/stockrl/web/dist/`에 생성되고 Python이 직접 제공합니다. 소스 수정 후에는 다시 빌드해야 합니다.

- `frontend/src/pages/`: 화면별 컴포넌트
- `frontend/src/components/ui.tsx`: 공통 버튼·토글·배지·카드·표·상태 표시
- `frontend/src/components/operations.tsx`: 공통 계좌·모델·체결·학습·자원 표시
- `frontend/src/api.ts`, `types.ts`, `hooks.ts`, `data.ts`: 공통 API·응답 타입·polling·표시 형식
- `frontend/src/styles.css`: 공통 스타일

기존 API와 hash 주소를 유지합니다. 공통 polling은 중복 요청을 합치고, 접힌 상세는 펼칠 때 렌더링합니다. 버튼은 요청 중·성공·실패를 표시합니다. [화면 기능 대조와 검증 방법](docs/frontend.md)

## GitHub와 로컬 파일

GitHub에는 소스·설정·설치/실행 스크립트·React 소스와 정적 빌드·필요한 expert 코드와 입력 자료를 전달합니다. 모델/PT, 설치된 Python 환경, Node 의존성, runtime의 계좌·DB·replay·로그·캐시, 컴퓨터별 인증 설정, 휴지통은 전달하지 않습니다.

DB를 올리지 않아도 설정과 스키마 코드로 첫 사용 시 새 DB가 생성됩니다. 새 PC에서 이전 학습·계좌를 이어가려면 worker를 정상 정지한 뒤 모델과 해당 runtime을 별도로 복사해야 합니다. 인증 정보는 새 PC에서 설정합니다.

모델 경로의 기준은 `src/stockrl/paths.py`, 설치는 `scripts/install_windows.py`, 시작은 `start_stockrl.py`입니다. 일반 프론트 수정에 Python 핵심 모델·계좌·학습 코드를 변경할 필요는 없습니다.

## 검증과 한계

React 전환 때 frontend 테스트 36개와 Python 정적 제공 테스트 3개가 통과했습니다. TypeScript typecheck와 Vite build, 주요 9개 화면·실제 API·polling·독립 제어, Python 단독 제공을 확인했습니다. GPU 잠금 수정 후 Champion과 전용 TradingMoE가 함께 실행되고 optimizer 횟수가 증가하는 것도 확인했습니다. 이는 해당 변경의 구동 확인이며 수익률 검증 결과가 아닙니다.

현재는 **PAPER 가상매매**입니다. 수수료·슬리피지·현금·보유량·NAV/P&L은 기존 paper engine이 계산합니다. 실주문은 사용자의 명시적 허용 전까지 OFF입니다. 외부 pretrained expert에는 각 원본 라이선스가 적용됩니다.
