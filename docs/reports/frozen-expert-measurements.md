# Frozen expert 독립 추론 측정

원본 가중치 고정, optimizer 없음. 배치 1~2종목, 과거 128시점, 미래 1시점의 CUDA 검사입니다.
RAM은 각 독립 프로세스의 Windows peak working set(패키지/CPU 적재 포함), VRAM은 PyTorch peak allocated입니다. 동시에 적재한 합계가 아닙니다.

| Expert | 실제 parameters | dtype | 원본 checkpoint MiB | 실질 가중치 MiB | RAM peak MiB | VRAM peak MiB | 입력 → 출력 | 추론 s |
|---|---:|---|---:|---:|---:|---:|---|---:|
| FinCast 1.0B | 991,436,960 | FP32 | 3,782.94 | 3,782.03 | 8,189.17 | 3,790.61 | {"series": [1, 128]} → [1, 1, 10] | 0.936 |
| EXAONE Finance | 202,319,520 | FP32 | 771.81 | 771.79 | 2,253.85 | 785.53 | {"series": [1, 128]} → [1, 21, 1] | 0.294 |
| Kronos Base + Tokenizer | 106,268,634 | FP32 | 405.41 | 405.38 | 1,379.69 | 424.27 | {"OHLCV_amount": [1, 128, 6]} → [1, 1, 6] | 0.418 |
| MarketGPT | 94,292,736 | FP32 | 5,999.99 | 359.70 | 1,287.32 | 370.78 | {"itch_tokens": [1, 32]} → [1, 1, 12160] | 0.191 |
| FinText Chronos Global | 46,154,240 | FP32 | 176.08 | 176.06 | 1,470.57 | 241.40 | {"series": [2, 128]} → [2, 8, 1] | 0.400 |
| FinText TimesFM Global | 19,809,304 | FP32 | 75.58 | 75.57 | 1,279.83 | 83.90 | {"series": [2, 128]} → [2, 1, 10] | 0.333 |
| Time-MoE Large | 453,196,800 | BF16 | 864.46 | 864.40 | 3,360.83 | 923.05 | {"series": [1, 128]} → [1, 1] | 0.393 |
| Toto 2.0 | 312,684,608 | FP32 | 1,192.82 | 1,192.80 | 3,079.32 | 1,330.89 | {"series": [2, 128]} → [9, 1, 2, 1] | 0.532 |
| MacroHFT slope/1 | 45,507 | FP32 | 0.18 | 0.17 | 893.92 | 8.31 | {"single_state": [1, 36], "trend_state": [1, 9], "previous_action": [1]} → [1, 2] | 0.126 |
| MacroHFT slope/2 | 45,507 | FP32 | 0.18 | 0.17 | 893.63 | 8.31 | {"single_state": [1, 36], "trend_state": [1, 9], "previous_action": [1]} → [1, 2] | 0.128 |
| MacroHFT slope/3 | 45,507 | FP32 | 0.18 | 0.17 | 894.13 | 8.31 | {"single_state": [1, 36], "trend_state": [1, 9], "previous_action": [1]} → [1, 2] | 0.125 |
| MacroHFT vol/1 | 45,507 | FP32 | 0.18 | 0.17 | 894.21 | 8.31 | {"single_state": [1, 36], "trend_state": [1, 9], "previous_action": [1]} → [1, 2] | 0.124 |
| MacroHFT vol/2 | 45,507 | FP32 | 0.18 | 0.17 | 894.12 | 8.31 | {"single_state": [1, 36], "trend_state": [1, 9], "previous_action": [1]} → [1, 2] | 0.117 |
| MacroHFT vol/3 | 45,507 | FP32 | 0.18 | 0.17 | 893.81 | 8.31 | {"single_state": [1, 36], "trend_state": [1, 9], "previous_action": [1]} → [1, 2] | 0.125 |

합계: 2,226,435,844 parameters; 실질 가중치 7.450 GiB; 펼친 원본 checkpoint 12.959 GiB.
모델·zip·공식 source·격리 Python 환경 등을 포함한 artifact 디렉터리 파일 크기 합계: 15.315 GiB (filesystem 압축/공유 블록 사용량과는 다릅니다).

## 검사 범위와 제한

- MarketGPT는 실제 ITCH 체결 데이터가 없는 상태의 native vocabulary 합성 토큰 검사, MacroHFT는 합성 36+9 feature 검사입니다. 수익성 검증이 아닙니다.
- 나머지는 원본 예제의 일별 초과수익률 또는 프로젝트의 실제 일봉 입력으로 형상·유한 출력·고정 가중치를 확인했습니다. 정확도 backtest가 아닙니다.
- Kronos의 amount가 없는 입력은 missing 표시와 원본 허용 방식의 0값을 사용했습니다. 실제 매수금액이 관측되었다는 뜻이 아닙니다.
- Kronos Base 실제 parameters 102,310,592 + tokenizer 3,958,042 = 106,268,634. Buffer/공유 tensor를 parameter에 중복 합산하지 않습니다.
- MarketGPT 원본 pt는 6,291,444,790 bytes. 5,159,780,352 bytes의 결정적 causal mask와 optimizer 등이 포함됩니다. 원본 파일은 보존하고 inference에서 mask만 재생성하여 learned parameter 94,292,736개를 사용합니다. 원본 zip 1,054,913,634 bytes는 별도 보존합니다.
- Toto는 native network AST와 strict checkpoint load를 유지하고 Lightning을 불필요하게 가져오는 GluonTS bridge import/class만 실행에서 제외합니다. 원본 source/checkpoint 파일은 수정하지 않습니다.
- Time-MoE의 이름 200M은 활성 parameters 규모입니다. 실제 총 parameters는 453,196,800개입니다. BF16 원본을 그대로 사용합니다.
- Toto의 runtime buffer도 존재합니다. 실질 가중치 memory와 실제 VRAM peak를 구분합니다. RAM/VRAM peak는 입력 크기와 package 환경에 따라 달라집니다.
- MacroHFT는 공개된 ETHUSDT 하위 정책 6개만 확인되었습니다. 학습된 상위 hyper-agent checkpoint는 제공되지 않았습니다.
- EarnHFT/EarnMore/DeepScalper/EIIE: 코드만 확보, 확인한 공식 public 경로에 trained checkpoint 없음. random 초기화를 pretrained policy로 표시하지 않습니다.
- EXAONE은 연구 license, MacroHFT/EarnHFT는 확인한 저장소에 LICENSE 없음. EIIE 원본은 GPL-3.0의 legacy TensorFlow 구현입니다.

## 통합 설계

모든 expert를 independent frozen 모듈로 유지합니다. capability와 memory 예산으로 top-k를 선택하고 단일 GPU에서 순차 실행 후 worker를 종료하여 VRAM을 반환합니다.
원본 출력 단위·분위수·시간축·symbols·as_of를 보존한 typed evidence → modality projection → cross attention → BUY/HOLD/SELL·비중·현금·가치 head 구조입니다. 예측 결과를 평균하거나 서로 다른 가중치를 합치지 않습니다.
이번에는 학습하지 않으므로 공통 trading head/router는 학습된 정책이 아닙니다. head의 실행 가능한 매매 출력은 차단합니다. 실시간 Champion/Candidate·replay·paper account와 연결하지 않습니다.
단일 상위 checkpoint는 pinned original artifacts + router/adapter/fusion 상태 + hash manifest로 구성할 수 있습니다. 현재는 manifest가 원본 파일을 참조하며 하나의 self-contained giant weight 파일로 복사하지 않습니다.
능력 중복: Chronos/TimesFM은 동일 초과수익률 영역, FinCast/EXAONE/Time-MoE/Toto는 시계열 예측 영역이 겹칩니다. 현재 제거하지 않습니다. 향후 같은 입력의 사용 기록/출력 상관관계를 측정한 뒤에만 후보로 판정합니다.
worker 시작마다 적재 비용이 발생하므로 표의 추론 시간은 cold-load 전체 시간이 아닙니다. serving 최적화 시 원본 능력을 검증한 뒤 dependency별 persistent worker를 고려합니다.

## 공식 원본과 고정 revision

- FinCast 1.0B: [Vincent05R/FinCast](https://huggingface.co/Vincent05R/FinCast/tree/2d7d90b159db8961d27c2cf165d51195902ef92b), [vincent05r/FinCast-fts](https://github.com/vincent05r/FinCast-fts/tree/488b19d1d85fa2b3d4b93469530cefdcf1cc97a4)
- EXAONE Finance: [LG-AI-Research/EXAONE-Forecast-for-Finance-1.0](https://huggingface.co/LG-AI-Research/EXAONE-Forecast-for-Finance-1.0/tree/e9d7c4e8dcbbb5d0b559c41fddeb409dac152c53), [LGAI-Research/EXAONE-Forecast](https://github.com/LGAI-Research/EXAONE-Forecast/tree/5e6e5224f2a0205f146bc772082d7fa4f59bd7c5)
- Kronos Base + Tokenizer: [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base/tree/2b554741eca47781b64468546e77fef3e85130e6), [NeoQuasar/Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/tree/0e0117387f39004a9016484a186a908917e22426), [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a)
- MarketGPT: [aaronwheeler/MarketGPT-100m](https://huggingface.co/aaronwheeler/MarketGPT-100m/tree/f80057ebcacbd43dbf72c2d7acd804e58eeaccd4), [aaron-wheeler/MarketGPT](https://github.com/aaron-wheeler/MarketGPT/tree/070c3ad6d10877fcb9651ba9f7affefdbf68320c)
- FinText Chronos Global: [FinText/Chronos_Small_2023_Global](https://huggingface.co/FinText/Chronos_Small_2023_Global/tree/74a9ce6eb66854cf406c8642ab18ec9aea7ae468), [DeepIntoStreams/TSFM_Finance](https://github.com/DeepIntoStreams/TSFM_Finance/tree/7a377ee8e9a0f6bb49af6c79252839d0efeebe4c)
- FinText TimesFM Global: [FinText/TimesFM_20M_2023_Global](https://huggingface.co/FinText/TimesFM_20M_2023_Global/tree/22801e12efb5a6f11df5dad5f8886fb0b478c3a1), [DeepIntoStreams/TSFM_Finance](https://github.com/DeepIntoStreams/TSFM_Finance/tree/7a377ee8e9a0f6bb49af6c79252839d0efeebe4c), [google-research/timesfm](https://github.com/google-research/timesfm/tree/e77303c5513c84aa8b32639851c3f7f92f59645e)
- Time-MoE Large: [Maple728/TimeMoE-200M](https://huggingface.co/Maple728/TimeMoE-200M/tree/794591bfeb1225fdf742cec0f4c71f20c3f3b87e), [Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE/tree/915bfda4c78a544d62a2bec6ab22948423059236)
- Toto 2.0: [Datadog/Toto-2.0-313m](https://huggingface.co/Datadog/Toto-2.0-313m/tree/a7bab288f5e95f8606f8306f86659357e1c001ef), [DataDog/toto](https://github.com/DataDog/toto/tree/9213d2c86f716c43010a0a3b2b6a0e3f440e6743)
- MacroHFT slope/1: [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54)
- MacroHFT slope/2: [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54)
- MacroHFT slope/3: [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54)
- MacroHFT vol/1: [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54)
- MacroHFT vol/2: [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54)
- MacroHFT vol/3: [ZONG0004/MacroHFT](https://github.com/ZONG0004/MacroHFT/tree/31e5ef41f93b2aea6e63e6ae2267675c997a8e54)
