# 주식 pretrained policy 편입 결과

2026-10-03 실제 실행 기준. 이번 추가 작업에서는 학습과 optimizer update를 실행하지 않았다.

## 최종 모델

- `C:/Users/hushm/Desktop/모델/TradingMoE.pt`
- 독립 expert **20개**: 기존 시장 분석 8개 + MacroHFT 6개 + 신규 주식 정책 6개.
- adapter/controller 포함 총 **2,229,175,352 parameter = 2.229175352B**.
- 실제 파일 **8,021,807,613 bytes = 8.021808 GB**.
- 기존 파일 백업: `TradingMoE.before-stock-policies.pt`.
- 기존 14개 가중치 hash와 기존 controller parameter 보존. 기존 optimizer moment는 parameter 이름으로 새 그룹에 연결하며 누적 update **2,605회**를 유지한다.
- 기존 원본 checkpoint와 새로 다운로드한 zip/pth/scaler/config 파일을 보존했다. 서로 다른 expert 가중치는 merge하지 않았다.
- 실행 중지 상태를 유지한다. 다음 TradingMoE 시작 시 확장된 PT를 한 번만 읽는다.

## 실제 로드한 pretrained weights

추가 정책은 모두 FP32이다. 아래 parameter 수는 native policy 전체이며 SAC의 critic/target critic도 포함한다.

| Expert | 파일 | 실제 bytes | Parameter | 입력 shape | 원본 action | 학습 universe |
|---|---|---:|---:|---|---|---|
| FinRL-DAPO-SR | model_rl.pth | 3,295,286 | 822,952 | [1,1009] | 84개 signed share action | NASDAQ-100 데이터의 실제 84종목 |
| Adilbai | final_model.zip | 4,875,406 | 393,669 | [1,3008] | [action type, position fraction] | AAPL, AMZN, GOOGL, MSFT, TSLA |
| RLTradingAgent | ppo_trader.zip | 276,393 | 20,228 | [1,90] | 0=SELL / 1=HOLD / 2=BUY | MSFT |
| FinRL bot A2C | agent_a2c.zip | 283,439 | 31,523 | [1,171] | 17개 signed share action | 아래 17종목 |
| FinRL bot PPO | agent_ppo.zip | 412,897 | 31,523 | [1,171] | 17개 signed share action | 아래 17종목 |
| FinRL bot SAC | agent_sac.zip | 5,135,576 | 576,294 | [1,171] | 17개 signed share action | 아래 17종목 |

Adilbai의 `scaler.pkl`은 2,423 bytes, `config.json`은 602 bytes다. 원본 파일을 보존하고 scaler의 실제 mean/scale/feature 순서와 config를 PT 내부에 포함했다. 다시 fit하지 않는다.

FinRL bot의 native tensor 종목 순서:

`AAPL, AMD, AMZN, CAT, CRWD, GOOGL, GS, HD, IBM, INTC, META, MSFT, NVDA, PYPL, T, TSLA, V`

DAPO의 native tensor 종목 순서:

`AAPL, ADBE, ADI, ADP, ADSK, AEP, ALGN, AMAT, AMD, AMGN, AMZN, ANSS, ASML, AVGO, AZN, BIIB, BKNG, BKR, CDNS, CHTR, CMCSA, COST, CPRT, CSCO, CSGP, CSX, CTAS, CTSH, DLTR, DXCM, EA, EBAY, ENPH, EXC, FANG, FAST, FTNT, GILD, GOOG, GOOGL, HON, IDXX, ILMN, INTC, INTU, ISRG, KDP, KLAC, LRCX, LULU, MAR, MCHP, MDLZ, MELI, META, MNST, MRVL, MSFT, MU, NFLX, NVDA, NXPI, ODFL, ON, ORLY, PANW, PAYX, PCAR, PEP, QCOM, REGN, ROST, SBUX, SIRI, SNPS, TMUS, TSLA, TXN, VRSK, VRTX, WBA, WBD, WDAY, XEL`

## 원본 입력과 공통 출력

공통 출력은 다음과 같다. 별도의 `raw_policy_output`과 shape에 원본 action 배열을 보존한다.

```text
symbol_id / buy_score / hold_score / sell_score / target_weight / confidence
```

- DAPO: cash + 84 prices + 84 share quantities + 8×84 indicators + 84 sentiment + 84 risk = 1,009. 결측 뉴스 점수는 저자 코드의 neutral 3 규칙을 재사용한다. checkpoint에는 Gaussian actor만 있다. activation 정보가 weight에 없으므로 원본 학습 생성자의 Tanh default를 사용한다. 별도 backtest 코드의 ReLU 생성자로 바꾸지 않는다.
- FinRL: cash + 17 prices + 17 share quantities + 8×17 indicators = 171. 실제 원본 indicator 컬럼을 사용하며 OHLCV만 제공되면 FinRL과 같은 StockDataFrame 방식으로 계산한다. signed action을 비중으로 오해하지 않고 DAPO 100주/FinRL 20주 hmax, 매도 후 매수 순서, 현금과 수수료를 반영해 목표 비중으로 변환한다.
- Adilbai: 원본 계산식의 60×50 feature를 저장된 scaler로 transform하고 계좌 상태 8개를 붙인다. action type은 원본 환경처럼 정수 truncation으로 0=HOLD/1=BUY/2=SELL을 해석하고 position fraction으로 주문 수량을 변환한다. 원본 관측 코드의 scaled-Close 계좌 convention도 보존했다.
- MSFT PPO: 현재 실행일을 제외한 직전 15일×6 feature를 사용한다. 원본 discrete action 확률을 공통 BUY/HOLD/SELL 순서로 변환한다.
- confidence는 native action 확률 또는 continuous action strength다. 수익이 날 확률을 추정한 값이 아니다.

실행 입력은 `snapshot.stock_policy_history`의 날짜·종목·OHLCV/native indicator 기록과 `snapshot.policy_account`의 cash/NAV/positions다. `snapshot.expert_inputs`에서 expert별 정확한 native observation을 전달하는 경로도 유지한다.

## Router와 학습 연결

```text
시장 expert 8개 → market state
→ stock / MacroHFT policy evidence
→ 종목 applicability mask + trainable policy router
→ policy attention fusion → 기존 controller
→ BUY/HOLD/SELL + target weight
```

- 주식 정책을 시장 분석 expert 목록에 섞지 않았다. policy 계층의 독립 registered submodule로 등록했다.
- MSFT에서는 적용 가능한 주식 정책들을 함께 선택한다. 초기 router는 가능한 의견에 같은 비중을 주며 이후 학습으로 변경할 수 있다.
- A005930과 ETHUSDT에는 NASDAQ/MSFT 정책을 적용하지 않는다. ETH는 기존 native MacroHFT 경로를 유지한다.
- 포트폴리오 정책은 정확한 순서의 전체 84/17종목 입력이 필요하다. 일부 종목이나 native feature가 없으면 해당 출력은 unavailable/masked다. 합성 관측값이나 crypto 입력으로 채우지 않는다.
- 적용 가능한 주식 정책이 없으면 기존 시장 expert/controller 경로를 유지한다.
- optimizer 그룹은 기존 `adapter`, `controller_router_fusion`을 사용한다. 새 EvidenceAdapter, policy projection, policy router가 이 그룹에 포함된다. 20개 native expert는 모두 frozen이다.
- 기존 전용 continuous worker는 계속 ETH 과거 구간을 사용한다. 신규 주식 정책은 정확한 주식 history/account가 주어질 때 활성화된다. 이번 결과가 실시간 주식 feed 전체 연결을 의미하지는 않는다.

## Fresh process 실제 결과

- `TradingMoE.pt` 로드 **20.900초**.
- 추가 정책 **6/6 CUDA 추론 성공**. 원본을 직접 로드한 실행과 raw output이 일치했다.
- Python audit hook으로 원본 weight/scaler 파일 접근을 차단한 상태에서도 PT 하나에서 6개가 추론됐다.
- 신규 6개 weight hash와 기존 14개 weight hash가 모두 일치했다.
- 기존 optimizer moment의 개수와 새 optimizer load를 확인했으며 누적 update 2,605회를 보존했다.
- 개별 정책, combined MSFT, unavailable mask에서 controller 출력은 모두 finite였다.
- 동일 시점 **2023-01-03** 실제 주식 기록으로 MSFT에 6개 주식 expert가 참여했다. ETHUSDT/A005930에는 주식 policy mask가 전부 false였다.
- combined MSFT native 정책 실행/router/fusion/controller **0.052520초**. PT cold load와 observation 준비는 제외한 시간이다.
- 이 구간 controller 결과는 MSFT `HOLD`, target weight `0.016389251`이었다. ETHUSDT와 A005930은 `HOLD`, target weight 0이었다. 수익성 검증이나 새 학습은 수행하지 않았다.

전체 native 출력, common 출력, hash, universe, shape, 시각과 계측값:

```text
C:/Users/hushm/OneDrive/문서/ChatGPT/금융매매모델/artifacts/experts/stock-policies/native-verification.json
C:/Users/hushm/OneDrive/문서/ChatGPT/금융매매모델/artifacts/experts/stock-policies/native-inputs.json
C:/Users/hushm/OneDrive/문서/ChatGPT/금융매매모델/artifacts/experts/stock-policies/fresh-process-report.json
C:/Users/hushm/OneDrive/문서/ChatGPT/금융매매모델/artifacts/experts/stock-policies/downloads.json
```

## 다시 실행

기존 expert 전용 venv Python을 사용한다.

```text
scripts/add_stock_policy_experts.py download
scripts/add_stock_policy_experts.py verify
scripts/add_stock_policy_experts.py package
scripts/add_stock_policy_experts.py fresh
```

`package`는 중복 편입을 거절한다. `publish`는 큰 PT를 로드하지 않고 기존 전문가 registry의 표시 정보를 갱신한다. registry에는 기존 14개를 포함한 20개가 표시된다.

## 공식 출처와 고정 revision

- [FinRL-DAPO-SR](https://github.com/Ruijian-Zha/FinRL-DAPO-SR): `02bdc24912c740183bfb9a0af11078d207ecba8b`
- [DAPO 실제 checkpoint](https://huggingface.co/rz2689/finrl-dapo-grpo-sentiment-risk): `c8510f5efffc6074cba6958990ef603d2742152f`
- [Adilbai weights/scaler/config/원본 preprocessing](https://huggingface.co/Adilbai/stock-trading-rl-agent): `a317fec1939eda44d04088b47b48ca3ee158bc1f`
- [RLTradingAgent](https://github.com/maksimprivalov/RLTradingAgent): `4cb10fa1d127014184bfea6912a2d01b981bff86`
- [FinRL trading bot](https://github.com/tpeiqi/FinRL_trading_bot): `9094922335f26fce6a4dd3bd9c3d58601819ea03`
- [공식 DAPO NASDAQ 데이터](https://huggingface.co/datasets/benstaf/nasdaq_2013_2023): `b80bc15e4320eac68f53cfdd2fff3365e55dfedd`
