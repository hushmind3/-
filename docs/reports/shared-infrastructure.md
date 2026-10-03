# TradingMoE 공용 부품 분리

## 현재 세 계층

- 현행 모델: TradingMoE, native 시장/매매정책 expert, adapter/router/fusion/controller.
- 공용 인프라: 시장 패널, Experience, reward, paper account, replay store.
- 기존 모델 실행 지원: GlobalMarketTransformer, OnlineGlobalAgent, online의 learner/validation/checkpoint.

| 기능 | 유일한 구현 |
| --- | --- |
| Panel, 17차원 feature, 16차원 시장 context, stable ID | `src/stockrl/market_panel.py` |
| Experience, REWARD_VERSION/DEFINITION, Horizon, MarketObservation, 증분 CSV | `src/stockrl/experience.py` |
| 실제 체결 후 계좌/종목 손익, 지연 보상, 휴장 시점 정산 | `src/stockrl/rewards.py` |

기존 `global_transformer`/`online.data`/`online.rewards` 경로는 저장된 Experience와 기존 명령을 위해 같은 객체를 import한다. Panel/Experience/reward 구현을 두 벌로 보관하지 않는다. 현행 MoE 실행기와 bridge는 공용 모듈을 직접 사용한다.

기존 Transformer/agent의 실행 지원과 HTTP endpoint, 모델 family 분기는 유지했다. 모델 실행 지원 제거는 별도 선택 사항이며 이번 변경에서 제거하지 않았다.

## CLI

`cli.py` 최상위는 argparse/json/os/Path/paths만 import한다. `live-feed` 시작이 torch/core/기존 Transformer/agent를 먼저 읽지 않는다. 해당 모델이 필요한 명령은 함수 내부에서 import한다. `train`은 이미 입구에서 실행을 거부하므로 도달할 수 없던 옛 훈련 본문을 외부에 보존하고 제거했다. 같은 오류 메시지로 계속 거부한다.

## 유지한 계약과 검증

Experience dataclass, 증분 CSV, horizon, MarketObservation, 모든 reward 함수 본문은 분리 전 AST와 일치한다. Panel의 feature/context 순서는 그대로이고 context tuple의 중복을 없앴다. tensor window를 호출할 때만 torch를 import한다.

회귀 항목 여섯 개: 기존 import의 객체 동일성, 옛 pickle class 참조, horizon, 구형 모델/torch를 읽지 않는 공용 데이터 경로, 가벼운 CLI/help/중지된 명령, 실제 feed/TradingMoE import의 독립성.

전체 Python 테스트는 수정 완료 후 한 번 실행했다. **167개 통과, 18.062초**, 신규 회귀 여섯 개 포함. 결과는 `runtime/code-cleanup/shared-tests.log`에 보존한다. 새 checkpoint 적재나 장시간 모델 학습은 검증에 포함하지 않는다.

변경 전 소스는 외부 `<프로젝트 이름>-휴지통/공용부분리-*`에 보존했다. 모델 파일·계좌·replay·UI·자동조립 실행 상태는 이동하거나 초기화하지 않았다.
