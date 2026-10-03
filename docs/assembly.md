# 조립 · 자동실험

현재 main은 ff655b1의 20개 expert 편입을 포함한다. 조립은 그 구현과 이후 UI/저장 경로 정리를 유지한다.

## 실행과 저장

화면은 /#assembly, API는 /api/assembly/status, start, stop, generate, next, settings, trial/start, trial/stop.

- 기존 Candidate TradingMoELifecycle 슬롯을 사용한다. 기존 Candidate가 실행 중이면 충돌을 표시하며 두 번째 worker를 만들지 않는다.
- 공용 champion.pt는 읽기 전용이다. Candidate마다 파일을 복사하지 않는다.
- recipe와 router/fusion/controller/adapter state만 runtime/assembly 아래에 저장한다. 실제 기본 state는 약 3.37 MiB이다.
- state.json, queue.json, current_recipe.json, champion_recipe.json, history.jsonl, recipes/, results/, trainable/에 진행상태·조건·성적을 보존한다.
- 탈락/승격 후보의 시험 계좌와 replay 작업파일은 휴지통/assembly-experiments로 이동한다. recipe, 작은 state와 성적·이유는 남는다.
- 웹 재시작은 동일 Candidate worker에 재연결한다. 정지 후 재개도 저장된 recipe/계좌를 사용한다.

## 재사용

Expert Registry의 실제 id, 원본 hash/revision, role, universe, native input shape를 읽는다. 프론트에 전문가 이름/목록을 하드코딩하지 않는다.
TradingMoE의 EvidenceAdapter, VerticalController, native MacroHFT preprocessing, FairGpuScheduler와 registry_owner를 재사용한다.
실행 제어는 기존 TradingMoELifecycle 및 Supervisor Candidate 슬롯을 사용한다.
체결·수수료·slippage·NAV·손익·보상·replay는 TradingMoEPaper → PaperAccount → 기존 reward engine/GlobalReplayBuffer를 호출한다.
online/validation.py의 Transformer 모델 복사·승격 경로는 변경하거나 호출하지 않는다.

## 추가한 부분

assembly_orchestrator.py: 영속적인 소규모 mutation, 대기열, Registry 변경 감지, 시험 단계, recipe 승격/탈락.
run_assembly_trial.py: 공용 base 하나에서 같은 시점/비용/초기 자금의 Champion/Candidate 비교.
assembly.js: 전광판, 실제 Registry 조립표, 독립 조작, 시험 성적, 대기열과 이력.

## 평가 조건과 범위

실제 native 출력 journal의 최근 24개 시점을 사용한다. 앞 구간 replay 예선과 뒤 구간 paper 비교는 겹치지 않는다.
8개 대형 시장 expert의 원본 출력 cache를 사용한다. 캐시 timestamp는 변경하지 않으며 refresh 제한을 넘는 값은 제외한다.
MacroHFT는 공식 ETHUSDT의 36+9 특징과 각 시험계좌 previous_action으로 native Q를 다시 계산한다. GPU 동시 전문가 제한은 1개다.
주식 policy universe와 원본 관측 규격은 바꾸지 않는다. ETH 구간의 주식 정책은 입력 없음으로 mask된다.
시험에서는 탐험/optimizer를 끄고 고정 조건으로 비교한다. 운영 Champion의 학습은 기존 경로를 유지하며 승격 recipe는 다음 모델 시작부터 적용한다.
replay 성과가 Champion보다 0.10%p 이상 나쁘면 예선 탈락한다. paper 승격은 비용 차감 수익 양수, Champion 초과, 손실폭 악화 1%p 이내일 때만 가능하다. 동점은 승격하지 않는다.
첫 실행의 24시점에서 양쪽이 HOLD했다면 거래 0/수익 0으로 기록하며 성능 개선이라고 주장하지 않는다.

새 expert는 Registry 파일 변경에서 감지되어 NEW 및 다음 mutation 풀에 들어간다. 새로운 body가 아직 공용 PT에 포함되지 않았다면 시험은 명시적으로 보류된다. 없는 가중치·native 입력을 만들어내지 않는다.

## 확인 내역

자동 후보 생성 → 실제 replay/paper 비교 → 탈락 → 다음 recipe 자동 장착을 실행했다.
후보별 3,536,863 byte 작은 state를 저장했으며 8GB base 복제/수정은 하지 않았다.
생성/교체/재시작/Registry JSON 변경/독립 설정 4개 테스트와 기존 paper 연결 7개 테스트를 통과했다.
