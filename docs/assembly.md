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
frontend/src/pages/Assembly.tsx: 전광판, 실제 Registry 조립표, 독립 조작, 시험 성적, 대기열과 이력.

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
생성/교체/재시작/Registry JSON 변경/독립 설정/승격 스위치 6개 테스트와 기존 paper 연결 7개 테스트를 통과했다.
실제 시험 4개가 탈락하고 다음 recipe가 자동 장착됐다. 마지막 비교는 양쪽 각각 replay 8회 + paper 16회 판단, NAV 10,000 USD → 10,000 USD, 체결 0건/보상 0이었다. 동점을 승격으로 처리하지 않았다.
시험 정지·재개 시 계좌와 누적 판단 수·실행시간·최대 손실폭을 함께 보존한다. 전체 정지는 자동조립 예약도 중지한다.

## 이력 기반 후보 생성

`AssemblyOrchestrator.generate()`는 기존 공용 PT와 Candidate 슬롯을 그대로 사용한다.

- recipe fingerprint는 base hash, expert revision/조합/역할/universe, market/policy routing, 활성 시장 expert의 cache 유효기간, controller 종류를 정렬해서 SHA256으로 만든다. ID·시간·성적·파일 경로, 평가에서 사용하지 않는 policy refresh/cache interval은 제외한다.
- history.jsonl, recipes/*.json, 현재 후보·대기 후보와 비교해 이미 생성/평가한 fingerprint를 반복하지 않는다. 이전 recipe는 현재 등록 revision으로 정규화하며 옛 ON/OFF·router·refresh 설명에서도 변경 내용을 복원한다.
- 새 expert의 revision별 상태는 state.json의 expert_trials에 untested/tested/promoted/rejected로 남긴다. 첫 probe 생성 시 candidate_id를 예약해 대기열에서도 반복하지 않는다. 탈락 후 다른 mutation으로 진행하며 revision이 바뀌면 새 probe를 허용한다. 기존 new_experts/API 형식은 유지한다.
- ETH 평가에서 native 입력이 없는 주식 전용 정책은 구조에 남지만 ON/OFF 탐색에서 제외한다. 시장 cache refresh는 실제 평가 cache 나이가 threshold를 넘어 사용 여부가 달라질 때만 탐색한다. MacroHFT는 매 bar 재계산하므로 policy refresh mutation은 만들지 않는다.
- 정책 top-k와 temperature, 시장 router, expert ON/OFF를 실제 탐색한다. 유효 expert 전체를 선택하는 top-k끼리의 변경이나 top-k=1 상태의 temperature 변경은 만들지 않는다.
- 동일 candidate의 journal/archived score는 한 번만 집계한다. paper 비교가 있으면 우선 사용하고 없으면 replay delta를 쓴다. mutation·expert·mutation family별 개선/악화 횟수와 평균 delta를 저장/표시한다. 복합 변경에는 기여도를 나눠 배분하므로 인과 효과라는 주장은 하지 않는다.
- 신뢰도는 `min(고유 window 표본/256, 1) × min(고유 window 수/8, 1)`이다. 같은 24시점 결과가 반복되어도 표본 신뢰도는 늘리지 않는다. 누락된 표본 수에는 신뢰도를 주지 않는다. 선택 가중치는 0.25~4로 제한하고, 새 설정에는 같은 family/expert 통계를 약하게 전달한다.
- 비-probe 선택의 25%는 균등 탐색으로 남긴다. 성적이 좋았던 서로 다른 변경 2개를 조합하는 경로는 최대 15%의 시도로 제한한다. 한 단계 후보가 소진되면 최대 64개의 임의 두 변경 조합만 살펴본다. 유효한 미시험 조합을 못 찾으면 반복 대신 명시적으로 알린다.
- recipe에는 fingerprint, mutation_operations, generation_reason, selection_weights, exploration_probability, probed_experts를 보존한다. 화면에는 생성 이유를 표시하고 상세에는 expert_trials/history_stats를 제공한다. 기존 조립 endpoint와 요청은 바꾸지 않는다.

평가 worker는 기존 24시점 실행과 체결 엔진을 유지하고, 실제 market cache 나이와 평가 종목을 result의 evaluation_context로 전달한다. 이 개선은 평가 기간을 늘리거나 수익성을 입증한 것이 아니다.

검증: 조립 24개 + 기존 paper 연결 7개 테스트. 중복·재시작·탈락 후 진행·revision 변경·시험/승격 상태·정책 routing·무효 변경 제외·history 확률 반영·짧은 반복 window 신뢰도·성공 변경 조합·소진 처리를 확인한다. 실제 controller forward 테스트는 작은 CPU 모듈만 사용하며 expert PT를 복제하거나 다시 검증하지 않는다.
