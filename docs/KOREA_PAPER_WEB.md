# 한국 시장 운영 안내

현재 프로젝트 규칙은 루트 `AGENTS.md`와 `README.md`가 기준입니다. 이 문서는 한국 market 설정을 실행·확인하는 짧은 안내입니다.

## 실행과 상태 확인

Windows에서 `서버켜기.cmd`을 실행합니다. 기본 로컬 대시보드는 `http://127.0.0.1:8766/`, 상태 API는 `http://127.0.0.1:8766/api/status`입니다. API에서 feed와 agent 진행 시각, replay 건수, Candidate 학습·검증 단계, paper 계좌, 실제 주문 OFF를 확인합니다. API가 응답하지 않으면 현재 운영 상태는 미확인입니다.

한국 market runtime은 `runtime/markets/korea/live`에 저장합니다. feed 파일, replay SQLite, paper 계좌, cursor, 로그와 검증 상태를 포함하며 Git에는 넣지 않습니다. Champion과 Candidate 가중치는 바탕화면 `모델` 폴더에만 둡니다.

## 가상매매와 학습

Champion은 운영 판단과 Champion paper 계좌를 담당합니다. Candidate는 replay batch로 학습합니다. 별도의 Candidate 관찰용 paper 계좌는 최근 학습 완료 가중치의 행동을 보여주며 학습 보상·승급 점수에 들어가지 않습니다.

승급은 고정된 Candidate snapshot과 Champion을 서로 다른 paper 계좌로 같은 미래 bar, 초기 자금, 비용 조건에서 비교합니다. 현재 구현은 128개 미래 bar로 평가하며, Candidate 순손익률이 더 높고 비교 중 Champion이 바뀌지 않았을 때만 승격합니다. 검증 snapshot은 RAM에 유지하고 별도 checkpoint 파일을 만들지 않습니다.

키움 자격 증명은 운영체제 자격 증명 보관함을 사용합니다. provider 연결 또는 인증 성공은 주문 허용을 뜻하지 않습니다. 실제 주문은 기본 OFF입니다.