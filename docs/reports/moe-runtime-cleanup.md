# MoE 운영 상태 정리 — 2026-10-03

## 실행 상태

- 운영 전광판은 접히지 않는다. Feed, 모델, 키움 수신, optimizer를 같은 화면에 표시한다.
- 별도 MoE worker의 `loaded/status`가 모델 실행 판단의 기준이다. 예전 공통 agent의 정지 상태로 MoE를 정지라고 표시하지 않는다.
- 기존 1,215,508 판단 건수를 현재 모델 실적으로 표시하던 코드를 제거했다.
- 공식 ETHUSDT 과거 데이터의 시점과 현재 worker/학습 완료 시간을 구별한다. 과거 데이터에는 연도를 표시한다.
- optimizer 누적 횟수, 화면 확인 이후 증가량, loss, reward, 실제 학습 완료 시각을 표시한다.
- MoE 실행 시 경험 학습 화면도 MoE의 별도 DB를 표시한다. 예전 공통 통계를 섞지 않는다.

## Replay

이전 공통 replay 516,669,440 bytes를 `휴지통/2026-10-03-legacy-replay`로 이동했다. 가상계좌와 모델 가중치, 원본 시세는 초기화하지 않았다.

MoE replay 앞부분에 evidence context가 없는 경험이 있어 FIFO의 뒤쪽 정상 경험을 막았다. `pending_batch(timestamps=...)`는 LIMIT을 적용하기 전에 사용할 수 있는 문맥을 가진 경험을 고른다. 이전 기록을 자동 삭제하지 않는다.

checkpoint 저장 전에는 업데이트한 경험을 삭제하지 않는다. `remaining_for_update`는 아직 업데이트하지 않은 경험, `awaiting_checkpoint`는 업데이트했지만 모델 저장을 기다리는 경험이다. 저장 성공 후 기존 acknowledge 경로가 완료 경험을 정리한다.

## 전문가 적용

최종 controller에서 stock policy universe를 다시 검사하던 중복 gate를 제거했다. adapter가 실제 출력이 존재하는 symbol의 coverage를 제공한다. 원본 portfolio policy의 입력 차원과 종목 순서는 native adapter에서 유지한다. 적용 가능한 policy 출력이 없는 symbol도 범용 market/controller 경로로 처리할 수 있다.

## 확인

- paper bridge 테스트 7개 통과. 새 회귀 테스트는 문맥이 없는 오래된 경험이 정상 경험을 막지 않으며 기존 행이 보존되는 것을 확인한다.
- 실제 worker optimizer: 2,605 → 3,708 (+1,103), 저장 완료 후 같은 계좌로 재시작.
- UI 중복 3개 카드와 지연 계산 제거. 이전 화면/코드 사본은 `휴지통/2026-10-03-moe-ui-residue`에 보관한다.
