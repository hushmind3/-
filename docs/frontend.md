# React 프론트엔드

## 구조

- `frontend/src/App.tsx`: 기존 9개 hash 화면과 상단 연결 상태.
- `frontend/src/pages/`: 운영/시장, 경험 학습, 승급전, 연결 설정, 상세 기록, 전문가, TradingMoE, 조립 실험.
- `frontend/src/components/ui.tsx`: 버튼, 토글, 배지, 카드, 통계 카드, 패널, 표, 로딩/오류/빈 상태, 명령 진행/성공/실패.
- `frontend/src/components/operations.tsx`: 모델 카드, 통화별 계좌, 포지션, 체결, 학습 전광판, replay, GPU/runtime 카드.
- `frontend/src/api.ts`, `types.ts`, `data.ts`, `hooks.ts`: API, 응답 타입, 표시 형식, endpoint별 공통 polling.
- `frontend/src/styles.css`: 공통 토큰과 컴포넌트 스타일. 페이지별 중복 카드/버튼 CSS 없음.
- `src/stockrl/web/dist/`: Vite 정적 빌드. Python이 직접 제공하는 유일한 운영 프론트.

## 개발/운영

Node LTS는 UI 개발 및 빌드에만 필요합니다.

```powershell
cd frontend
npm ci
npm run dev
npm run typecheck
npm test
npm run build
```

개발 화면은 `http://127.0.0.1:5173/`이며 `/api` 요청을 기존 Python의 8766 포트로 전달합니다.
빌드 후 기존 `서버켜기.cmd`로 운영합니다. Python 웹서버만 실행하면 됩니다. Node/Vite 운영 프로세스가 없습니다.
정적 build는 프로젝트와 Python package data에 포함합니다. 모델/PT/replay/실행 상태/Node 의존성은 포함하지 않습니다.
UI 소스 수정은 해당 페이지와 공통 컴포넌트만 읽으면 됩니다. DOM ID, innerHTML, 전역 화면 렌더링 함수는 없습니다.

## 기존 기능 대조

| 기존 화면/hash | React 화면 | 보존한 기능 |
| --- | --- | --- |
| `#control` | `Operations.tsx` | 운영 상태, 모델별 optimizer/loss/reward, 승급 진행, replay, 독립 판단/체결/학습 스위치, feed 시작/재연결, 웹만 재시작, 전체 정지, 두 장기계좌 초기화, Champion/Candidate 독립 시작/저장 후 정지, 통화별 계좌/보유종목/체결/비용 |
| `#markets` | `Operations.tsx`의 Markets | 시장별 수신과 선택, 전체시장/검색/최근5분 필터, 시세/거래량/판단/목표비중, 호가/체결 입력 |
| `#learning` | `Learning.tsx` | 학습 상태/업데이트/샘플/시간, 결과대기/학습가능/완료/보류, DB, 일자별 처리, 보상, 목표, 시간봉별 입력/실제 샘플, 상세 모델/학습 수치 |
| `#promotionTrial` | `Trial.tsx` | 고정본 비교, 진행/버전/동일조건, 별도 시험계좌/포지션/체결/비용, 하루 시험/승격 기록, 장기 운영계좌와 분리 |
| `#connection` | `Connection.tsx` | 실전/모의 시세 환경, 계좌/키 입력, 인증 연결, 저장된 인증 재확인, 공인IP, 연결 상태 |
| `#system` | `Details.tsx` | GPU/RAM/모델/runtime 상태, 스케줄러, 학습/추론/저장 시간, 로그, 오류, DB/출력/비용/종목확장 |
| `#experts` | `Experts.tsx` | 실제 Registry, 합계/역할/dtype/메모리/적재/활성/고정/router/최근사용/추론시간, 입력/출력shape, 원본 raw 조회와 JSON 다운로드, 통합 결과 조회/다운로드, 진단/불가 원인 |
| `#trading-moe` | `TradingMoE.tsx` | 전용 시작/저장 후 정지, PID/적재/장치, 계좌/ETH 단위 변환/체결, 현재 판단/수직계층, NAV/P&L/reward/loss/replay/optimizer |
| `#assembly` | `Assembly.tsx` | 자동조립 시작/정지, 후보생성, 시험 시작/정지, 탈락/다음, 자동교체/승격/감지 스위치, 동적 Registry/NEW, recipe/예선/paper비교/queue/history, PT복제0/작은state |

`#overview`, `#market`, `#trial`, `#details`, `#portfolio`, `#promotion` 별칭도 유지합니다.
기존 Python endpoint와 요청 payload를 사용합니다. 모델/학습/승급/계좌 구현을 복제하지 않았습니다.

## 갱신/명령

- 공통 endpoint당 하나의 요청/타이머를 공유하고 겹친 요청을 합칩니다.
- 상태 5초, TradingMoE 1초, 조립 3초. 비활성 탭에서는 주기 요청을 멈춥니다.
- 화면을 떠나면 해당 전용 endpoint의 polling/요청을 해제합니다.
- API 명령 진행 중 중복 클릭 차단. 성공 뒤 실제 상태를 다시 조회합니다.
- 실패하면 ON/OFF를 바꾸지 않고 오류를 표시합니다. 이전 데이터 표시와 연결 오류를 구분합니다.
- React reconciliation으로 바뀐 DOM만 갱신합니다. 접힌 상세는 내부 내용을 mount하지 않습니다.
- 과거 loss/계좌/판단은 기록으로 표시합니다. 정지 모델의 이전 RAM/VRAM을 현재 적재량으로 표시하지 않습니다.

## 검증

`frontend/tests/`: 9개 화면, hash, 필터, 공통 컴포넌트, 명령/오류/중복, 모든 제어 endpoint/payload, 원본 출력, 실제 응답 형식, polling 공유/중단/복구.
API 변경 버튼을 자동 테스트에서는 mock하므로 계좌 초기화나 키 교체를 수행하지 않습니다.
브라우저에서는 실제 Python API 조회와 독립 모드/조립 설정 토글을 변경 후 복원하고 페이지 전환, raw/fusion 조회, polling을 확인합니다.
Python 정적 제공과 경로 탈출 차단은 `tests/test_react_assets.py`로 확인합니다.
기존 모델 lifecycle/체결/학습은 backend 구현을 유지하며 프론트 테스트에서 대용량 모델을 재훈련하지 않습니다.
React 전환은 먼저 로컬에서 완료·검증했으며, 이후 사용자의 업로드 요청에 따라 GitHub main에 반영했습니다.
