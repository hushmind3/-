# StockRL 금융매매 모델

이 프로젝트의 목표는 특정 trader나 고정 전략을 영구 모방하는 것이 아니라, 시장 경험과 비용 차감 순손익으로 정책을 발전시키는 자율 트레이딩 에이전트다. 모델은 시장·종목·시간축·포트폴리오 상태를 바탕으로 종목 선택, BUY/HOLD/SELL, 자금 배분, 포지션 유지·교체·청산을 학습한다.

수수료, 거래세, 슬리피지, 스프레드, 현금, 보유수량, 평단, 실현·평가손익, 유동성, 자본 규모는 환경이 제공한다. 실제 매매전략은 시장 경험으로 모델이 발견하는 것이 목표다. 공개 모델, teacher와 과거 trader 기록은 초기 금융 문법을 익히는 교육 재료이며 최종 정책이 아니다.

운영 루프: `시장 관찰 → 판단 → 가상체결 → 가상계좌 순손익 계산 → 결과가 성숙한 경험을 runtime의 bounded SQLite replay에 저장 → candidate 학습 → champion과 같은 미학습 구간의 paper-account 순손익 비교 → 개선 시에만 승격`. 결과를 기다리는 `pending` 경험은 현재 메모리에만 있어 재시작 후 복구되지 않는다. 추론과 candidate 학습은 분리되어 시장 관측을 막지 않는다. 실제 주문은 기본 OFF이며 사용자가 명시적으로 허용하기 전까지 실행하지 않는다.

## 현재 파일 배치

- 이 프로젝트 폴더에는 소스 코드, 설정, 문서와 정적 연구 데이터가 있다.
- Windows 모델 폴더 `C:\Users\hushm\Desktop\모델`에는 `champion.pt`와 `candidate.pt`만 둔다.
- runtime은 프로젝트 폴더의 `runtime/markets/<market>/live`에 둔다. 현재 한국 운영 데이터는 `runtime/markets/korea/live`를 사용한다. NASDAQ 운영을 추가하면 `runtime/markets/nasdaq/live`를 쓴다. 기본 launcher 경로는 프로젝트 안이지만, `web --runtime` 명시 인자에는 아직 경계 검사 누락이 있다(아래 제한 항목 참조). 이 비공개 저장소에는 복구를 위한 runtime snapshot을 포함하고, 모델 가중치(`.pt`, `.pth`, `.ckpt`, `.safetensors`)는 제외한다. 실행 중 변경된 runtime 자료는 이후 GitHub 저장 시점의 snapshot으로 반영된다.
- `StockRL Start.bat`은 `STOCKRL_MARKET=korea`로 시작한다. `scripts/run_global_paper.ps1`에는 `-Market nasdaq`처럼 시장 이름을 줄 수 있다. 시장별 시세 설정도 해당 시장 설정 파일로 지정해야 한다.
- replay와 가상계좌 등 실제 한국 runtime은 `runtime/markets/korea/live`에 있으며 비공개 GitHub 복구 snapshot에 포함한다. 과거 커밋 `613c330`에는 이전 위치의 runtime snapshot도 남아 있다.
- 이전 위치 `%LOCALAPPDATA%\StockRL\runtime-global-korea-live`도 아직 남아 있다(확인 시 22개 파일, 43,960,537 bytes). 현재 feed/agent는 이 폴더를 사용하지 않고 프로젝트 runtime을 사용한다. 기존 폴더 삭제는 자동 도구 검토가 거부해 미완료이며, 삭제 완료로 간주하지 않는다.
- `web --runtime` 명령행 인자는 프로젝트 경계 검사를 우회할 수 있다. 이 인자로 프로젝트 밖 경로를 실행하지 말 것; CLI 경로 검사 보완 전에는 기본값 또는 프로젝트 내부 경로만 사용한다.
- 시세 CSV가 64MB를 넘으면 최근 512개 시각과 참고시장별 오래된 봉 20개만 남긴다. 중복 방지 기록도 최근 8일만 둔다.
- 실패해 쓰지 못한 경험은 재시도용으로 남고, 미사용 replay는 100,000개 한도 안에서 관리한다. 검증 경험은 최근 64개 검증 시각까지만 유지한다.
- 웹 실행기는 설정상 새 경험 약 4,096개마다 candidate 학습을 시도한다. 설정은 batch 1, optimizer update 1이다. 재시작 후 bounded SQLite replay, cursor, paper 계좌와 저장된 validation ledger는 남는다. 결과 미성숙 `pending` 목록은 메모리뿐이라 복구되지 않는다.
- 현재 비교기는 같은 검증 bar를 champion과 candidate의 별도 paper account에 순차 적용해 비용 차감 순손익을 비교한다. 보호 champion SHA 불일치로 candidate 학습과 승격은 현재 보류 상태다.
- candidate가 기각되면 champion 복사본으로 초기화해 다음 학습을 시작한다.
- 현재 champion 파일 SHA256은 사용자가 보호 대상으로 지정한 기준 SHA256과 다르다. 계보가 확인될 때까지 64개 bar 검증이 끝나도 candidate 승급과 다음 candidate 학습을 보류한다.

## 초기 연구 / 부트스트랩 기록

초기에는 가중 teacher 앙상블, GRU actor-critic, teacher 지식 증류, 소규모 PPO 미세 조정을 연구했다. 행동은 **SELL / HOLD / BUY**이며, critic의 스칼라 출력은 상태 가치 추정치다. 이 구현은 현재 프로젝트의 최종 학습 목표가 아니라 연구·비교용 baseline이다.

> 이 저장소는 연구·엔지니어링 프로젝트이며 수익을 보장하거나 투자 조언을 제공하지 않는다. 포함된 휴리스틱 정책은 실행 가능한 시연용 teacher이며 공개 사전학습 시장 모델이 아니다. 일반적인 주식매매 체크포인트가 이 프로젝트의 관측 구조와 행동 의미에 맞는다고 가정할 수 없다. 로컬 Stable-Baselines3 PPO/A2C/DQN 체크포인트는 명시적 adapter를 통해 사용할 수 있다. 사용 전 입력 특징, 행동 의미, 학습 출처를 확인한다. FinRL은 트레이딩 에이전트 학습용 오픈소스 프레임워크이며, 모든 프로젝트와 호환되는 사전학습 체크포인트 저장소는 아니다.

## 요구 사항 및 설치

Python 3.10 이상과 PyTorch 2.2 이상이 필요합니다. Apple Silicon에서는 arm64 환경을 만들고 사용하는 Python/macOS 조합에 맞는 공식 MPS 지원 PyTorch wheel을 설치하세요. `--device auto`는 MPS를 사용할 수 있으면 선택하고, 그렇지 않으면 CPU 또는 CUDA를 선택합니다.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell에서는 .venv\Scripts\Activate.ps1 실행
python -m pip install --upgrade pip
python -m pip install -e .
```

Stable-Baselines3 teacher도 사용하려면 다음을 설치하세요.

```bash
python -m pip install -e '.[teachers]'
```

## 전체 실행 예시

```bash
python -m stockrl train --data data/sample_ohlcv.csv --epochs 2 --ppo-updates 1
python -m stockrl predict --data data/sample_ohlcv.csv
```

Windows PowerShell에서는 `python -m pip install -e .`을 실행한 뒤 `python -m stockrl ...`을 실행하세요. `baseline-checkpoints`에 저장된 출력은 과거 연구 기록이며 현재 live champion/candidate가 아닙니다. 운영 checkpoint는 모델 폴더의 `champion.pt`와 `candidate.pt`만 사용합니다. 검증 지표와 따로 떼어 둔 최종 테스트 백테스트 결과를 각각 출력합니다. train/validation/test는 시간순으로 70/15/15 비율로 나누며, 특징과 teacher 관측값에 미래 수익률을 사용하지 않습니다.

## 데이터 형식

CSV에는 `date,open,high,low,close,volume` 열이 필요합니다. 열 이름의 대소문자는 구분하지 않습니다. 포함된 작은 샘플은 기본 실행 확인용 합성 데이터입니다. 실제 데이터를 쓸 때는 가격 조정과 시각 정보가 올바른 장중 OHLCV 데이터를 사용하고, timestamp를 정렬·중복 제거하세요. 액면분할, 현실적인 슬리피지와 스프레드, 체결 방식, 거래 시간을 반영한 뒤 결과를 해석해야 합니다.

## Teacher 설정

`configs/teachers.json`을 수정하세요. 휴리스틱 teacher는 `trend`, `mean_reversion`, `volatility_aware`를 지원합니다. 0 이상인 `weight`가 앙상블 가중치입니다. 이 teacher들은 공통 3개 행동 logit과 스칼라 value를 출력합니다. 로컬 SB3 설정 예시는 다음과 같습니다.

```json
{"name":"local_ppo","type":"sb3","algorithm":"ppo","path":"models/ppo.zip","weight":1.5}
```

SB3 관측값은 펼친 window 특징 배열이며, 행동은 0=SELL, 1=HOLD, 2=BUY의 이산 인덱스여야 합니다. adapter는 행동 예측을 공통 logit으로 변환합니다. 이 정규화된 추론 API에서는 SB3 critic value를 사용할 수 없어 0으로 처리합니다. 다른 PyTorch 체크포인트는 `src/stockrl/core.py`의 `Teacher` 인터페이스로 연결하세요. teacher 특징·행동·value의 의미를 일관되게 정규화하는 것은 사용자의 책임입니다. teacher 평가 가중치는 명시적으로 설정하며, 이 프로젝트는 해당 가중치가 실험으로 검증됐다고 주장하지 않습니다.

## 명령어

```bash
python -m stockrl train --data PATH.csv --teachers configs/teachers.json --device auto
python -m stockrl predict --data PATH.csv --device auto
```

이 구현은 학습용 최소 PPO 루프이며 실전 주문 실행 엔진이 아닙니다. 실제 운영에 사용하려면 소스를 검토하고 거래·세금 규칙과 현실적인 시장 시뮬레이터를 추가하세요.

## 예전 GRU baseline 지속 관찰 기능 (현재 운영에 사용하지 않음)

아래 `continuous`와 `replay` 명령은 초기 GRU baseline의 예전 연구 기능이다. 이 명령들은 별도 replay/checkpoint 파일을 만들 수 있어 현재 운영 구조와 맞지 않는다. 현재 가상매매를 시작할 때는 웹 실행기만 사용한다. 현재 online agent는 성숙한 replay를 bounded SQLite에 보존하고 미성숙 outcome 대기 목록은 메모리에 둔다. validation ledger와 비교 계좌는 JSON으로 보존되며 validation 작업 queue는 메모리에서 동작한다. 모델 폴더에는 champion과 candidate만 둔다.

초기 모델을 한 번 만든 뒤, 툴/피드가 최신 봉을 계속 추가하는 CSV를 감시할 수 있습니다.

```bash
python -m stockrl continuous --data path/to/live_bars.csv
```

프로세스는 `Ctrl+C`까지 실행되며, 새 timestamp마다 판단을 기록하고 `runtime/latest_signal.json`을 원자적으로 갱신합니다. 신호의 `mode`는 항상 `paper`입니다. 이 JSON은 외부 자동매매 툴이 읽을 연동 경계입니다. 툴에서 사용자가 자동매매를 눌렀을 때 주문 실행은 그 툴이 담당하고, 모델 루프는 브로커 주문 API를 호출하지 않습니다. 실행 결과를 다시 학습하려면 툴이 `timestamp,action,reward` 열을 가진 CSV를 갱신하고 `--feedback path/to/feedback.csv`를 지정합니다. 실제 주문 체결과 손익 귀속은 툴에서 계산해 `reward`로 전달해야 합니다.

이하 내용은 당시 구현 동작을 설명하는 기록이다. 특히 `runtime/replay.pt` 출력 설명은 현재 실행 방식에 적용되지 않는다. 과거 데이터를 가상 체결로 흘려보내는 확인 명령:

```bash
python -m stockrl replay --data data/sample_ohlcv.csv
```

## 시장과 입력 호환성

보편적인 “모든 시장 호환” 기능은 없습니다. 미국/한국 주식, 선물, 가상자산 등에서도 **timestamp와 OHLCV 규약을 정규화한 바 데이터**는 공통 기반으로 처리할 수 있습니다. 시간대와 거래 세션, 액면분할/배당 조정, 종목 식별자, 가격 단위, 숏 가능 여부, 수수료·세금·슬리피지와 체결 모델은 공급자와 시뮬레이터 설정에 맞춰야 합니다. 기본 모델은 종목별 단일 시계열이므로 여러 종목/시장 혼합 학습에는 심볼/시장 식별 특징과 별도 검증이 필요합니다.

CSV 필수 열은 `date,open,high,low,close,volume`입니다. 선택 `bid,ask,bid_size,ask_size,buy_volume,sell_volume,trade_count` 열에서 호가 spread/imbalance와 체결 imbalance/intensity 특징을 만듭니다. 영상은 프레임 자체가 아니라 프레임 처리 단계가 추출한 수치로 입력합니다. 차트/거래량 영상에서 timestamp별 `video_chart_signal,video_volume_signal`을 추출해 CSV 행에 넣을 수 있게 열을 준비했습니다. 다른 트레이더 기록은 timestamp, 행동, 실현 손익을 시장 데이터와 정렬해 feedback CSV로 전달할 수 있습니다.

지속 루프는 파일에 새로운 timestamp가 추가되지 않으면 새 판단/학습을 하지 않습니다. 공급기는 확정된 봉을 append하거나 원자적 교체로 게시해야 합니다. 샘플 데이터만으로 실시간 피드, MPS 장기 실행, 외부 툴의 주문/피드백 연동을 검증할 수는 없습니다.

## 공개 teacher 체크포인트

`models/teachers/`에 공개 **학습 가중치 파일 6개**를 내려받아 저장했습니다. 모두 SB3/PyTorch로 불러와 Yahoo Finance의 실제 MSFT 일봉 데이터에서 추론했습니다. 출처마다 전처리와 관측 공간이 달라 모델별 adapter를 거쳐 공통 `SELL/HOLD/BUY` logits와 value로 변환합니다. action probability는 공개 모델이 보정한 확률이 아닙니다. 공개 정책이 선택한 행동에 큰 logit을 부여한 **증류용 pseudo target**입니다. PPO 모델의 value는 critic 출력, DeepBio DQN은 max Q입니다. Pfizer recurrent 모델은 공개 추론 인터페이스에서 critic 값을 가져올 수 없어 0으로 기록합니다.

| 공개 모델 / 파일 | 구조·실제 입력 | 원래 행동 → 공통 행동 | 라이선스·비고 |
|---|---|---|---|
| [maksimprivalov/RLTradingAgent](https://github.com/maksimprivalov/RLTradingAgent), `maksimprivalov_ppo_trader.zip` (276,393 B) | SB3 PPO MLP; 90개 실수 = 15×6 (`log_return,sma20,sma50,rsi14,macd,volume_change`) | 이산 행동 0 SELL, 1 HOLD, 2 BUY → 그대로 사용 | 저장소에서 LICENSE 파일을 찾지 못했습니다. 연구용 추론을 위해 내려받았습니다. 공개 README는 MSFT 일봉 OHLCV를 설명합니다. 실제 435행 출력 중 HOLD 433회, SELL 1회, BUY 1회로 편향이 확인돼 가중치를 1.0으로 낮춰 영향이 작은 teacher로 설정했습니다. |
| [jk2500/Pfizer-trader](https://github.com/jk2500/Pfizer-trader), `jk2500_pfizer_recurrentppo_lstm.zip` (28,532,063 B) | SB3-Contrib RecurrentPPO `MlpLstmPolicy`; (100,9) (`Open,High,Low,Close,Volume,Adj Close,MA10,MA50,RSI`) | 이산 행동 0 short, 1 long → SELL, BUY. 모델 자체에는 HOLD가 없습니다. | 저장소 archive에서 LICENSE 파일을 찾지 못했습니다. 저장소 설명에 따르면 PFE 데이터로 학습됐습니다. 별도 입력 scaler가 공개되지 않아 MSFT 앞 70%로 scaler를 fit한 뒤 전체 구간에 적용했습니다. LSTM hidden state는 봉 사이에 유지합니다. |
| [Adilbai/stock-trading-rl-agent](https://huggingface.co/Adilbai/stock-trading-rl-agent), `adilbai_final_model.zip` (4,875,406 B) + `adilbai_scaler.pkl` (2,423 B) | SB3 PPO MLP; 표준화 시장 특징 60×50개 + 포트폴리오 값 8개 = 실수 3,008개. scaler도 함께 받았습니다. | 실제 SB3 action space는 `Box([action_type, size])`입니다. 예측 첫 요소를 반올림해 0 HOLD, 1 BUY, 2 SELL로 바꾸고 size로 paper portfolio 상태를 갱신합니다. | Hugging Face model card의 라이선스는 MIT입니다. 카드에는 연속·이산 출력 설명이 섞여 있지만 checkpoint의 action space를 직접 읽어 Box(2)임을 확인했습니다. adapter는 이를 명시적으로 보수 변환합니다. 공개 성능 수치는 자체 보고라 검증된 성과로 취급하지 않습니다. |
| [deepbiolab/drl-trading](https://github.com/deepbiolab/drl-trading), `checkpoint_deepbiolab.pth` (13,668 B), `model_fold_1_deepbiolab.pth` (13,692 B), `model_fold_2_deepbiolab.pth` (13,692 B) | PyTorch Double-DQN Q MLP `3→64→32→8→3`; window 1의 정규화된 연속 차이값 (`Close,BB_upper,BB_lower`) | Q 행동 0 HOLD, 1 BUY, 2 SELL → HOLD, BUY, SELL | MIT 라이선스입니다. checkpoint 3개를 불러와 forward를 실행해 실제 state dict임을 확인했습니다. 공개 저장소는 AAPL 데이터와 기술지표를 설명합니다. 원본 normalizer artifact가 없어 MSFT 앞 70%로 열별 정규화값을 fit하고 이후 구간에 적용했습니다. |

**공개 프로젝트 4곳의 체크포인트 6개**를 연결했습니다. PPO 계열 2개, LSTM recurrent PPO 1개, DQN 3개입니다. 이 모델들을 HFT/scalping 전용이라고 볼 수는 없습니다. 확인된 출처의 모델은 일봉·단일 종목 또는 포트폴리오 정책이며, 초 단위 미만의 주문장 전략과는 다릅니다. 별도 공개 checkpoint를 찾지 못한 A2C 전용 모델은 등록하지 않았습니다.

다운로드 후 생성되는 출력:

- `outputs/public_teachers/<teacher>.csv`: MSFT의 timestamp 435개에 대해 모델별 행동, 세 행동 pseudo probability, value를 기록합니다.
- `outputs/public_teachers/ensemble.csv` 및 `distillation_targets.npz`: 여섯 teacher의 동일 가중 앙상블 target입니다.
- `checkpoints/public_distilled/final.pt`: 실제 MSFT 시계열 앞 70%로 distillation 2 epoch와 PPO update 1회를 수행해 만든 학생 체크포인트입니다.
- 테스트 구간(마지막 15%)의 단순 일봉 backtest: 이 실행의 누적 수익률은 **-34.11%**, 최대 낙폭은 **36.42%**였습니다. 모델 품질을 입증하는 결과가 아니라 실행 확인이며, 성과는 음수입니다.

실행 명령:

```bash
python -m pip install -e '.[teachers]'
python -m stockrl teacher-dataset --data data/msft_real_daily.csv --teachers configs/teachers.json --output outputs/public_teachers
python -m stockrl train --data data/msft_real_daily.csv --teachers configs/teachers.json --checkpoint-dir checkpoints/public_distilled --epochs 2 --ppo-updates 1 --device auto
```

### 내려받았지만 teacher로 사용하지 않은 공개 후보

- [xinghao2003/fyp Hugging Face dataset](https://huggingface.co/datasets/xinghao2003/fyp): dataset/model-results 설명은 있지만 공개 GitHub archive와 HF dataset 파일에서 직접 불러올 RL checkpoint를 찾지 못해 제외했습니다.
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL): PPO/A2C/DDPG 등의 학습 코드와 환경은 제공하지만, 이 프로젝트에 바로 연결할 공개 pretrained checkpoint 파일을 확인하지 못해 checkpoint teacher로 등록하지 않았습니다.
- A2C/HFT/scalping 키워드로 찾은 여러 저장소는 코드만 있거나 실제 weight가 없거나, action/observation 규약을 재구성할 파일이 없어 이번 실행 registry에서 제외했습니다. 공개 checkpoint를 확인하지 않은 알고리즘을 연결했다고 세지 않았습니다.

이번에 검토한 후보 중 라이선스 표기만을 이유로 내려받기·연결에서 제외한 모델은 없습니다. 위 두 저장소에서는 LICENSE 파일을 찾지 못했지만 요청된 연구 범위에서 checkpoint를 받아 실행했고, 그 상태를 manifest와 표에 표시했습니다. 저장소 archive에서 weight를 찾지 못했거나 실제 호환되지 않은 경우는 라이선스 문제와 구분해 제외했습니다.

실제 가격 파일 `data/msft_real_daily.csv`는 Yahoo Finance chart data에서 받아 2024-01-02부터 2025-09-25까지 유효한 일봉 435개로 정리했습니다. 합성 샘플이 아닌 실제 데이터지만 매매 빈도는 일봉입니다. teacher의 학습 종목·기간·정규화와 MSFT의 2024–2025 데이터가 달라 분포 이동이 크므로, 결과는 teacher 인터페이스의 실제 실행 확인으로만 해석해야 합니다. Adilbai scaler는 scikit-learn 1.2.2에서 pickle된 파일이라 현재 버전에서 호환 경고가 발생하지만 로드와 추론은 완료했습니다.

## 글로벌 지속 학습 Transformer 에이전트

기존 `TemporalActorCritic` GRU는 종목별 소형 비교 기준으로 유지합니다. 글로벌 정책은 `src/stockrl/global_transformer.py`의 시간축·시장축 교차 attention Transformer를 사용합니다. 기본 모델의 정확한 크기는 **511,848,836개 파라미터**입니다. width 1,408, head 16개, 교차 Transformer block 21개, 시장 특징 17개, 상품·시장·자산·시간 embedding, SELL/HOLD/BUY 정책 head와 value head로 구성됩니다. 설정을 `d_model=1792, n_layers=26, n_heads=16`으로 확장하면 약 10억 파라미터 규모가 됩니다. 실행 기본값을 소형으로 바꾸지 않습니다.

CUDA에서는 모델과 optimizer를 FP16으로 실행합니다. RTX 3070 8 GiB에서 측정한 학습 peak allocated VRAM은 6,391,110,656 bytes였고, 저장된 champion의 FP16 weight는 약 1.02 GB입니다. AdamW half-state를 쓸 때 epsilon을 1e-4로 설정해 NaN을 방지합니다. Apple Silicon에서는 기존 `device=auto` 경로가 MPS를 선택해 FP32로 실행합니다. 이번에는 실제 MPS 장치에서 실행을 검증하지 않았습니다.

### 글로벌 실제 데이터 받기

다음 명령은 Yahoo Finance chart의 실제 일봉 OHLCV를 `data/global_market_daily.csv`에 저장합니다. 실행 확인에 사용한 데이터는 2023-01-02부터 2026-09-25까지, 상품 52개, 행 47,860개입니다.

```powershell
python scripts/download_global_data.py --output data/global_market_daily.csv --start 2023-01-01
```

포함 상품은 한국 주식·지수, 미국 주식·지수·지수선물·섹터 ETF, 유럽·일본·홍콩·호주·인도 지수, 미국 국채금리, 채권 ETF, 통화쌍, 원유·천연가스·곡물·금·은·구리 선물, VIX와 상품 ETF입니다. 이는 글로벌 상품의 **일봉 가격 패널**이며 옵션 체인이나 실시간 호가·체결 피드는 아닙니다. adapter는 옵션·호가 입력 열을 지원하며, 실시간 또는 외부 provider가 아래 CSV 열을 제공하면 변환합니다.

필수 열은 `date,symbol,market,asset_class,open,high,low,close,volume`입니다. 일봉 데이터의 `date`는 날짜를, 실시간 feed는 UTC ISO timestamp를 사용합니다. 선택 열은 `bid,ask,bid_size,ask_size,buy_volume,sell_volume,trade_count,implied_volatility,open_interest,yield_change,days_to_expiry,video_chart_signal,video_volume_signal`입니다. 다른 trader 기록과 영상에서 추출한 신호도 이 특징 열 및 별도의 `date,symbol,action` teacher CSV로 정규화해 넣을 수 있습니다. 제공되지 않은 옵션·호가 값은 0으로 채워지므로 실제 관측값으로 취급하면 안 됩니다.

### GPU 설치 및 실행

Windows RTX GPU에서는 먼저 CUDA wheel을 설치하세요. 아래 명령은 이 환경에서 PyTorch 2.14.0 CUDA 13.2로 확인했습니다. Apple Silicon Mac에서는 CUDA 명령 대신 `python -m pip install -e '.[teachers]'`를 사용하세요.

```powershell
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e ".[teachers]"
python -m stockrl global-info --data data/global_market_daily.csv --device auto
```

실시간 feed 프로세스는 위 형식의 append-only CSV에 시장 bar를 timestamp별로 추가합니다. 아래 명령은 각 timestamp를 관찰·판단하고, horizon 이후 가상 손익에 수수료·slippage를 반영해 replay에 저장합니다. learner는 별도 thread에서 업데이트합니다. Ctrl+C로 종료합니다. CSV 연결은 feed adapter이며 거래소 구독기나 주문 API가 아닙니다.

프로젝트 루트에서 PowerShell로 다음 명령을 실행합니다. runtime 경로는 프로젝트 내부로 고정되어 있습니다.

```powershell
$runtimeDir = Join-Path (Get-Location) 'runtime/markets/korea'
python -m stockrl global-online --data "$runtimeDir/live/market.csv" --state-dir "$runtimeDir/live/agent" --model-dir "$env:USERPROFILE/Desktop/모델" --follow --poll-seconds 1 --device auto
```

공개 teacher 판단을 먼저 만들어 글로벌 에이전트의 imitation replay에 추가할 수 있습니다. 입력 파일에는 `date,action` 열이 있어야 하며 기본 teacher symbol은 MSFT입니다.

공개 teacher 판단은 runtime 폴더 안에 생성해 글로벌 에이전트에 연결할 수 있습니다.

```powershell
$runtimeDir = Join-Path (Get-Location) 'runtime/markets/korea'
python -m stockrl teacher-dataset --data data/msft_real_daily.csv --teachers configs/teachers.json --output "$runtimeDir/public-teachers"
python -m stockrl global-online --data "$runtimeDir/live/market.csv" --state-dir "$runtimeDir/live/agent" --model-dir "$env:USERPROFILE/Desktop/모델" --follow --teacher-decisions "$runtimeDir/public-teachers/ensemble.csv" --teacher-symbol MSFT --device auto
```

teacher 연결 확인에서는 public teacher ensemble 435행 중 시간순 학습 구간에 속한 404행을 imitation replay에 넣었습니다. 짧은 update 10회 중 2회는 teacher imitation, 8회는 paper 경험을 사용했고 모델 weight가 변경됐습니다. 산출물은 `runtime-global-teacher-final/metrics.json`입니다.

### 과거 온라인 학습 실험 기록

아래 경로와 `.pt` 산출물 목록은 당시 연구 실행 기록이다. 현재 운영 실행기는 이 파일들을 만들거나 읽지 않는다. 현재 모델 폴더에는 `champion.pt`, `candidate.pt`만 둔다. 현재 운영 replay는 bounded SQLite, paper 계좌와 validation ledger는 JSON, 미성숙 outcome 대기 목록과 validation 작업 queue는 메모리에 둔다.

아래 benchmark와 파일 목록은 당시 실험 산출물 기록이다. 지금의 저장 구조를 설명하지 않는다. 현재 경로와 저장 원칙은 README 상단의 `현재 파일 배치`를 따른다.

데이터는 시간순으로 나눴습니다. 앞 70%는 replay 학습, 다음 15%는 champion 검증, 마지막 15%는 별도 backtest에 사용했습니다. RTX 3070에서 합성 입력이 아닌 `data/global_market_daily.csv` 전체 과정으로 실행했습니다. 시장 상품 52개에서 관찰 10회, 판단 492건, 지연 outcome/reward 492건이 replay에 들어갔으며 replay 파일 크기는 9,005,607 bytes였습니다. update 1회 후 weight의 L1 변화량은 2,139.50이었습니다. candidate의 net validation score `0.000108`이 champion의 `-0.006814`보다 높아 이 실행에서는 승격됐고, 이전 champion은 `champion.previous.pt`에 보관했습니다. 다른 실측 실행에서는 candidate 점수가 낮아 자동 기각되고 champion이 유지됐습니다.

측정값은 inference p50 **29.5 ms** (cold-start 포함 p95 122.4 ms), batch 1의 1-step update **0.717 s**, peak VRAM **6.39 GB / 8.59 GB**, process peak RSS **2.46 GB**입니다. checkpoint 저장과 재로드에 성공했고 출력 shape은 `[1,52,3]`, parameter 수도 동일했습니다. 마지막 15% backtest는 symbol-time 판단 253건, 6단계에서 net return **-0.123%**, max drawdown **0.123%**였습니다. 짧은 smoke 구간 결과이며 전략 수익성의 증거가 아닙니다. CUDA 실행은 `torch==2.14.0+cu132`, GPU `NVIDIA GeForce RTX 3070`을 사용했습니다. 교체 기준은 수수료·slippage를 반영한 시간순 validation net return입니다.

산출물은 `runtime-global-verified/metrics.json`, `decisions.csv`, `backtest.csv`, `replay.pt`, `champion.pt`, `candidate.pt`입니다. follow CSV 모드는 `live_cursor.json`, 미결 가상거래 `live_pending.pt`, 현재 target position `live_positions.json`을 저장해 재시작 후 이어갑니다. 기본 replay capacity는 100,000건입니다. teacher가 없을 때는 최근 경험 50%, 오래된 경험 30%, 극단 변동·보상 경험 20%를 섞습니다. teacher가 있으면 10%를 teacher imitation 예제로 할당합니다. 새 경험 256건마다 후보를 업데이트하며 전체 과거 데이터를 epoch 단위로 다시 학습하지 않습니다. 과거 teacher row는 검증·테스트 누수를 막기 위해 historical run의 앞 70%에만 넣습니다.

모든 승격·기각은 연구용 paper 평가로만 취급합니다. 전용 데스크톱은 Yahoo chart와 Kraken 공개 OHLC 분봉에 연결합니다. 전체 호가창·옵션 체인 feed나 실제 broker 연결은 제공하지 않습니다. 실주문 경로는 broker adapter를 명시적으로 설치해야 사용할 수 있으며 기본값은 OFF입니다.

## 과거 데스크톱 제어판 구현 기록

이 절의 restart 후 replay/checkpoint 복구 설명은 과거 구현의 동작 기록이다. 현재 운영 경로에서는 replay와 pending 상태를 `.pt`로 복원하지 않는다.

고정된 0.5B Transformer와 기존 온라인 에이전트의 구조는 변경하지 않습니다. Windows 데스크톱 패널은 공개 feed 프로세스와 `global-online --follow` 프로세스를 관리합니다. 후자는 관찰 루프에서 champion 추론을 계속하고, 별도 learner thread가 candidate를 업데이트하고 검증합니다. GUI는 별도 프로세스이므로 GUI를 중지·재시작해도 replay, champion, cursor, 미확정 paper 결과, 검증 샘플, 포지션, 설정을 버리지 않습니다.

GUI 의존성을 한 번 설치한 뒤 다음 명령으로 패널을 실행하세요.

```powershell
python -m pip install -e ".[desktop]"
python -m stockrl desktop --device auto
# 또는 scripts\launch_global_agent.bat
```

기본 데스크톱 프로필은 `runtime-global-desktop/live`이며 Start를 누르면 공개 실시간 데이터 수집기를 시작합니다. `Historical real data (mock feed)`를 선택하면 `data/global_market_daily.csv`의 최근 행을 지정한 속도로 재생합니다. 패널에는 provider/process 상태, 종목별 최신 BUY/HOLD/SELL 확률과 value, champion checkpoint 시각, replay 건수, paper reward, candidate 상태, 승격·기각 횟수, CUDA 장치/VRAM, paper 포지션을 표시합니다. 같은 화면에서 시작, 정상 정지, 비상 정지를 할 수 있습니다. runtime 프로필은 `runtime-global-verified`와 분리되어 있습니다. 검증 모델은 첫 실행 때 새 데스크톱 프로필로 복사되며 원본은 덮어쓰지 않습니다.

수집기는 Yahoo Finance chart bar와 Kraken 공개 OHLC를 조회합니다. provider는 `MarketDataProviderAdapter.fetch(instrument)`을 구현하고 `register_market_data_provider`로 등록합니다. `STOCKRL_MARKET_PROVIDER_PLUGINS=module1,module2` 또는 여러 `--provider-plugin` 옵션으로 모듈을 추가할 수 있습니다. 확정된 bar를 표준 UTC timestamp와 함께 `--follow`가 읽는 CSV에 추가하고, 재시작 간 중복 방지를 위해 SQLite 고유 키 index를 사용합니다. provider 오류를 기록하고 실패한 종목은 대기 후 재시도합니다. Yahoo chart는 인증이 필요 없는 best-effort endpoint이며 보장된 streaming 서비스는 아닙니다. 설정된 시장은 한국·미국 주식, 주요 지수와 주가지수 선물, FX, 미국 국채 수익률 지수, 원유·금·은·구리 선물, VIX, BTC/ETH 등 36개 상품입니다. 공개 무료 feed는 일관된 실시간 옵션 체인·이력 서비스를 제공하지 않으므로 이후 adapter가 데이터를 제공하기 전까지 옵션 IV/open interest 값은 비어 있습니다. Yahoo와 Kraken 분봉만으로는 전체 호가창 깊이나 매수·매도 체결 구분을 얻을 수 없습니다. 데이터 제공 여부, 거래 시간, provider 요청 제한, 지연·오래된 bar는 상품마다 다릅니다. 요청 오류 후 재연결하지만 거래소가 닫혀 있을 때 bar를 만들 수는 없습니다. 확정 봉 guard를 추가한 뒤 Yahoo/Kraken 실제 확인에서 AAPL/BTC/ETH bar 2,397개를 provider 오류 없이 추가했고, 두 번째 조회에서는 중복 timestamp가 추가되지 않았습니다.

실행·정지 제어는 provider와 agent 프로세스를 함께 시작하거나 중지합니다. 정지 요청은 로컬 marker에 저장됩니다. learner는 진행 중인 update를 마친 뒤 agent가 champion, replay, 미결 경험, cursor, 포지션을 저장하고 provider가 SQLite index를 닫습니다. 하위 프로세스가 예기치 않게 종료되면 저장 파일에서 다시 시작합니다. `python -m stockrl desktop`을 다시 실행하면 동일 설정과 프로필을 엽니다. 기본 candidate update 간격은 replay 경험 256건이며 데스크톱 패널에서 바꿀 수 있습니다. candidate 학습은 feed 조회나 champion 추론을 막지 않습니다. 지연 reward는 해당 종목에서 설정한 개수만큼 bar를 관찰한 뒤 확정되므로, 거래가 없는 주식에 무관한 FX/crypto timestamp의 reward를 주지 않습니다. 승격에는 미사용 구간의 net validation score 개선이 필요하며 이전 champion은 `champion.previous.pt`로 저장합니다.

실주문 경로는 기본 OFF입니다. 이 프로젝트에는 실제 증권사 연결이 포함되어 있지 않습니다. broker plugin은 `stockrl.broker.BrokerAdapter`를 상속하고 `is_live = True`를 설정한 뒤 `supports_symbol`, `size_order`, `connect`, `place_order`, `emergency_stop`, `close`를 구현해야 합니다. 설정은 `STOCKRL_LIVE_BROKER_ADAPTER=module:Class`로 지정합니다. GUI에서는 예/아니요 확인과 `ENABLE LIVE ORDERS` 입력을 모두 요구합니다. 지원하지 않는 종목과 adapter가 위험 제한 수량을 0 이하로 반환한 주문은 전송하지 않습니다. 포함된 `MockBroker`는 실제 주문을 보내지 않습니다. 실전에서 broker를 연결하려면 상품별 주문 단위, 가격 정밀도, 위험 제한, 인증, 긴급 중지 기능을 broker 측에서 처리해야 합니다.

Windows에서 CUDA PyTorch 2.14.0+cu132와 RTX 3070으로 데스크톱·수집기를 확인했습니다. 첫 실행에서 공개 수집기는 설정된 36개 상품 전체에 대해 1분 OHLCV 행 26,015개를 가져왔습니다. 두 번째 실행은 새 행 2개를 기록했고 중복은 없었습니다. 모든 timestamp가 UTC로 해석됐고 `(symbol,date)` 중복도 0개였습니다. 이후 시장 CSV를 `global-online --follow`가 읽도록 했습니다. 관찰 14회에서 예측 28건, 지연 outcome 26건, replay 저장 22건이 발생했습니다. CUDA에서 candidate gradient update 1회가 실행됐고 weight L1 변화량은 3.503, update 시간은 8.00초였습니다. 준비된 상태의 champion inference p50은 약 22 ms였습니다. 모델 장치는 `cuda` / `NVIDIA GeForce RTX 3070`, peak allocated VRAM은 6.42 GB였습니다. 이 결과는 실시간 feed에서 agent까지 연결되는 paper 연구 흐름을 확인한 것으로 전략 수익성이나 주문 실행을 증명하지 않습니다. 앞서 기록한 과거 샘플 backtest의 held-out 수익률은 여전히 -0.123%입니다.

실제 과거 시장 행을 봉당 0.15초로 재생해 대시보드를 화면 없이 실행했습니다. timestamp 24개를 관찰하고 종목별 판단 1,180건을 표시했으며 outcome 1,136건을 확정하고 replay 예제 772개를 저장했습니다. learner는 candidate update 2회를 수행했지만 둘 다 검증에서 실패해 champion을 유지했습니다. candidate weight L1 변화량은 0.780과 3.867, 준비된 inference p50은 24.2 ms, candidate update p50은 0.385초, 관찰된 peak CUDA allocation은 6.39 GB였습니다. 짧은 smoke sample의 replay paper reward 합계는 -0.11%였습니다. UI에는 최신 판단 80행, replay 건수, GPU, candidate, 실주문 OFF 상태가 표시됐습니다. 정상 정지 후 GUI를 다시 시작하자 replay(772), update 횟수(2), 동일한 시장 cursor가 복원됐습니다. 판단은 1,181행으로 유지됐고 재시작한 mock feed는 중복 bar를 추가하지 않았습니다. 실행 파일은 `runtime-global-desktop-verified2/mock/agent/`에 있으며 `metrics.json`, `replay.pt`, `champion.pt`, `live_cursor.json`, `live_validation.pt`를 확인할 수 있습니다.

새 bar가 계속 들어오는 동안의 동시성을 확인하기 위해 더 느린 replay도 실행했습니다. 첫 candidate 완료 시점까지 agent는 timestamp 29개를 관찰하고 판단 1,447건을 생성했으며 outcome 1,395건을 확정하고 replay 예제 1,179개를 저장했습니다. `candidate_training`이 true인 동안 champion inference가 17회 더 실행됐고, 행이 추가되는 중에도 mock-feed 프로세스가 동작했습니다. 0.5B candidate는 update 1회(0.759초)를 완료하고 weight L1 3.589만큼 바뀐 뒤 검증 실패로 기각됐으며 champion 추론은 계속됐습니다. inference p50은 27.7 ms, peak CUDA allocation은 6.40 GB였고 정상 정지 상태에서 `candidate_training: false`를 확인했습니다. 파일은 `runtime-global-desktop-concurrency/mock/agent/`에 있습니다. 독립 MockBroker queue 확인은 `live_order: false`인 모의 체결을 반환했습니다. 실제 broker는 설정되어 있지 않으며 기본 provider에는 거래소 websocket 구독이나 전체 옵션 체인·호가창 feed가 없습니다.

## 브라우저 대시보드 (권장 실행)

Windows에서 한 번 실행하면 로컬 웹 대시보드, 공개 시세 수집기, 기존 champion 모델이 함께 시작되고 브라우저 창이 열립니다. 기본 실행은 주문 OFF paper 모드입니다.

```powershell
python -m pip install -e .
python -m stockrl web
```

또는 파일 탐색기에서 `StockRL Start.bat`을 실행하세요. 운영 대시보드 기준 포트는 `8766`입니다. 실행 상태와 replay는 시장별 프로젝트 runtime 경로(현재 `runtime/markets/korea/live`)에 저장하고 비공개 GitHub에 복구 snapshot으로 보관합니다. 모델 가중치 파일만 Git에서 제외합니다. 창에는 시세/모델 연결, champion, BUY/HOLD/SELL 판단, 가상 포지션, reward, replay 건수, candidate 승격/기각 기록, GPU/VRAM이 표시됩니다.

같은 PC의 휴대폰 브라우저에서 보려면 LAN 연결에서 `python -m stockrl web --host 0.0.0.0`로 실행하고 PC의 사설 LAN 주소를 여세요. 현재 웹판은 인증/HTTPS가 없으므로 인터넷에 직접 공개하거나 클라우드에 바로 배포하지 마세요. 클라우드 운영은 인증, HTTPS reverse proxy, persistent volume을 앞에 둬야 합니다. 실시간 데이터는 무료 공개 API의 best-effort 제공이며, 실제 브로커 주문 기능은 웹판에 연결하지 않았습니다.

간단한 health/status 확인: `GET /api/status`; 컨트롤: `POST /api/start` (`{"mode":"live"}` 또는 `{"mode":"mock"}`), `POST /api/stop`.

## 여러 시장 공용 학습 데이터 loader

`src/stockrl/market_training.py`은 여러 시장 데이터 출처를 위한 공용 loader를 정의합니다. 출처별 adapter가 UTC event/bar를 하나의 형식으로 정규화합니다. 공용 loader는 symbol ID, 시간순 window, 데이터 범위를 고려한 symbol sampling, 전체 시장 context, target return, 시간순 70/15/15 분할을 처리합니다. 정규화 adapter는 주식, ETF, 선물, 옵션, 금리, crypto를 지원합니다. 선택 필드에는 bid/ask, 거래 흐름, implied volatility, open interest 변화, yield 변화, 만기까지 남은 일수, 영상에서 추출한 차트·거래량 신호가 포함됩니다. 값이 없으면 기존 17개 특징 입력에 맞춰 처리합니다. 첫 출처 전용 adapter는 수정된 Nasdaq ITCH snapshot을 위한 `ITCHSnapshotAdapter`입니다.

loader는 연속된 instrument ID를 결정적으로 할당하고 종목별 노출량을 기록합니다. 노출이 적은 종목을 주로 sampling하고 일부는 거래량 가중 방식으로 뽑습니다. 각 sequence에는 현재 전체 시장에서 한 번 계산한 전역 context 통계 16개가 들어갑니다. 기존 0.5B backbone과 champion은 유지합니다. 격리된 candidate에는 0으로 초기화한 context residual과 기존 embedding으로 초기화한 확장 symbol embedding을 추가합니다. dry-run은 champion을 승격하거나 덮어쓰지 않습니다.

공식 규격에 맞춘 ITCH 5.0 재파싱이 끝나면 RTX 3070에서 CUDA dry-run과 64/128/256 symbol-group benchmark를 실행하세요.

```powershell
python -m pip install -e .
python scripts/bench_market_training_loader.py
```

결과는 `runtime-global-market-training/`에 저장됩니다. 디스크 기반 panel, symbol map, 전체 exposure CSV, 측정 지표, 격리된 update 1회 candidate가 포함됩니다. 이 명령은 장시간 학습을 시작하지 않으며 `runtime-global-cuda-final/champion.pt`에 쓰지 않습니다.
### RTX 3070 dry-run 결과 (2026-09-26)

수정된 ITCH snapshot 전체 CSV를 공용 adapter와 loader로 불러왔습니다. 관측 시장 규모는 **instrument 8,694개 / UTC 시간 행 49,446개**였으며, memory-mapped 공용 panel은 **16,765,457,436 bytes (15.62 GiB)**를 차지합니다. 시간 분할은 엄격한 순서대로 train `2019-01-30 09:00:00`–`19:08:45 UTC`, validation `19:08:46`–`21:12:28 UTC`, test `21:12:29`–`2019-01-31 01:00:00 UTC`입니다. 입력은 sequence length 128에서 종목별 특징 17개와 전체 시장 context 특징 16개를 사용합니다.

NVIDIA GeForce RTX 3070에서 CUDA forward/backward/update 1단계를 측정했습니다.

| 샘플당 종목 수 | 입력 텐서 | 단계 시간 | 초당 종목-window 수 | 최대 할당 VRAM | 프로세스 RSS |
|---:|---|---:|---:|---:|---:|
| 64 | `[1,128,64,17]` | 1.269 s | 50.42 | 2.43 GiB | 3.39 GiB |
| 128 | `[1,128,128,17]` | 2.256 s | 56.74 | 2.95 GiB | 3.50 GiB |
| 256 | `[1,128,256,17]` | 4.126 s | 62.04 | 4.82 GiB | 3.85 GiB |

기록된 process peak RSS는 4.88 GiB였습니다. panel 생성 파일은 디스크에 memory-mapped 상태로 남습니다. 8,694개 instrument의 직접 ID는 모두 고유합니다. 이전 8,192-way ticker hash를 사용하면 이 시장에서 ID 3,357개가 충돌합니다. 짧게 설계된 dry-run의 세 benchmark batch에서는 서로 다른 instrument 575개를 선택했고, 8,119개는 노출이 0이었습니다. 따라서 전체 시장 coverage는 장기 sampling·학습으로 쌓아야 하며, 이 세 batch가 보장했다고 주장하지 않습니다.

격리 candidate는 optimizer update를 마치고 저장·재로드됐습니다. 재로드 출력은 유한값이었고 logits은 `[1,128,3]`, values는 `[1,128]`이었습니다. 30초 held-out validation minibatch에는 reward를 계산할 수 있는 instrument 11개가 있었습니다. 기대 net return 평균은 champion `-0.00128370`, candidate `-0.00128072`였습니다. 이 작은 minibatch 하나만으로는 개선을 입증할 수 **없으므로** candidate를 승격하지 않았습니다. 기존 champion의 SHA-256 `4100da96158c777426008f6c1a874951de160c17e4a5566f7dda4743a3c36128`은 유지됐으며 파일도 덮어쓰지 않았습니다. 결과는 `runtime-global-market-training/dryrun_metrics.json`, 종목별 exposure는 `runtime-global-market-training/symbol_exposure.csv`에 저장됩니다.

이 결과는 수정된 ITCH snapshot으로 loader와 제한된 CUDA dry-run을 확인한 것입니다. 광범위한 시장 학습이나 수익성 있는 정책을 증명하지 않습니다. 포함된 정규화 source adapter는 다른 시장 feed가 사용하는 공통 진입점입니다. 여러 시장을 지원한다고 주장하려면 provider별 parser와 더 긴 다중 국면 학습·평가를 추가해야 합니다.

## Runtime snapshot status — 2026-09-29 KST

- 마지막 runtime 포함 GitHub 커밋은 `c98ef4c`이며 16:05:57 KST에 저장됐다.
- live 실행이 이어져 16:07:48 KST 확인 때 다음 tracked 파일이 snapshot 이후 다시 바뀌어 로컬 변경으로 남았다: `agent/metrics.json`, `agent/replay.sqlite3`, `live_feed_metrics.json`, `logs/feed.log`, `market.csv`, `market.csv.sqlite3`.
- SQLite 두 파일은 원본 복사 없이 read-only 온라인 백업으로 확인했다. 둘 다 `PRAGMA integrity_check=ok`였지만 현재 백업 내용은 GitHub snapshot과 달랐다. 현재 replay 백업 13,438,976 bytes, market DB 백업 5,808,128 bytes; GitHub market DB snapshot은 5,767,168 bytes다. 따라서 GitHub는 `c98ef4c` 시점 복구본이며 그 이후 live 기록은 로컬에 남아 있다.

## 미성숙 경험의 재시작 처리

- `global_online.follow_csv()`의 미성숙 판단 목록 `pending`과 포트폴리오 목록 `portfolio_pending`은 메모리에서 빈 목록으로 시작한다. cursor(`live_cursor.json`), paper 계좌(`paper_account.json`), 이미 성숙해 SQLite replay에 들어간 경험은 저장되지만, 아직 결과가 나오지 않은 판단 경험 목록은 저장·복구되지 않는다.
- 재시작 전에 미성숙 경험이 있었고 cursor가 그 판단 bar를 지난 상태라면 이후 결과가 해당 경험에 연결되지 않을 가능성이 있다. 실제로 몇 건이 손실됐는지는 확인된 기록이 없어 단정하지 않는다.

- 현재 확인된 파일 상태: Desktop `모델`에는 `champion.pt`, `candidate.pt`만 있고 프로젝트 및 `runtime/markets/korea` 안에는 `.pt`가 없다. 운영 launcher는 두 운영 checkpoint를 Desktop 모델 폴더에서 사용한다. 별도 연구 도구는 실행 시 다른 runtime 경로에 candidate/replay checkpoint를 만들 수 있으므로, 현재 2개 파일 원칙 아래에서는 `warmstart_*`, `pretrain_portfolio_agent.py`, `distill_*`, `continuous`, `replay`, `train`을 운영 도구로 실행하지 않는다. 이 저장 코드 경로는 실제 생성 파일이 현재 없다는 사실과 별개다.
