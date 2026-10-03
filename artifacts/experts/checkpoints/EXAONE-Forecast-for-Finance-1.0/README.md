---
license: other
license_name: exaone
license_link: LICENSE
pipeline_tag: time-series-forecasting
library_name: pytorch
tags:
  - time-series
  - time-series-forecasting
  - foundation-model
  - finance
  - attention-free
  - zero-shot
  - exaone
---

# EXAONE Forecast for Finance

**EXAONE Forecast for Finance** is a time series foundation model (TSFM) built for financial
forecasting. It replaces self-attention with two linear-time operators — a causal
1D convolution for temporal mixing and a group-aware pooling MLP for variate
mixing — so cost grows linearly in both sequence length and variate count. A
masked-context augmentation makes it robust to the missing spans that are
pervasive in financial data, and it is pretrained on a synthetic financial corpus
whose generator reproduces heavy tails, volatility clustering, jumps, regime
shifts, and cross-asset dependence.

This repository holds the **model weights**. The **inference code** lives at
[LGAI-Research/EXAONE-Forecast](https://github.com/LGAI-Research/EXAONE-Forecast)
and is released under a different license — see [License](#license).

- 📄 Technical report: [arXiv:2609.04239](https://arxiv.org/abs/2609.04239) (also bundled here as [`EXAONE_Finance_v1.0_Technical_Report.pdf`](./EXAONE_Finance_v1.0_Technical_Report.pdf))
- 💻 Code: https://github.com/LGAI-Research/EXAONE-Forecast

## Overview

How EXAONE Forecast for Finance differs from other time series foundation models, in the two
choices that set a TSFM's cost and its inductive bias — how it mixes information
along time, and along variates — and in what it was pretrained on.

| Model | Temporal mixing | Variate mixing | Financial-domain |
|---|---|---|---|
| Chronos-2, Moirai, Moirai-MoE, Toto, UniTS | Attention | Attention | ❌ |
| Chronos, Chronos-Bolt, TimesFM, Lag-Llama, TimeGPT, Timer, MOMENT, Sundial, TEMPO, ROSE, Time-MoE, PatchTST-FM, Kairos, YingLong, CleanTS, TabPFN-TS, VisionTS | Attention | — | ❌ |
| TiRex (xLSTM), TempoPFN, FlowState, Reverso | RNN / SSM | — | ❌ |
| TTM (TinyTimeMixer) | MLP | MLP | ❌ |
| **EXAONE Forecast for Finance (Ours)** | **CNN** | **MLP** | ✅ |

EXAONE Forecast for Finance is the only **attention-free convolutional** TSFM in this
comparison, and the only one pretrained on **financial-domain data**, which we
generate synthetically. Both of its mixers are linear-time, so cost grows
linearly in sequence length and in variate count rather than quadratically.
A dash (—) means the model has no variate-mixing operator and forecasts each
series independently.

<sub>CNN: convolutional neural network · RNN: recurrent neural network ·
SSM: state-space model · MLP: multi-layer perceptron</sub>

## Model Configuration

| | |
|---|---|
| Parameters | 202M |
| Encoder blocks | 12 |
| Hidden dimension / FFN | 1024 / 4096 |
| Temporal mixing | Causal 1D CNN — 512 channels, kernel 7, 2 conv layers per block |
| Variate mixing | Group-aware pooling MLP — hidden dimension 512 |
| Patch size (input = output) | 16 |
| Context length | 512 |
| Maximum horizon | 2048 |
| Output | 21 quantile levels (0.01, 0.05, 0.10, …, 0.95, 0.99) |
| Attention layers | none |
| Precision | float32 |

## Evaluation Results

Zero-shot results on **FinVerse**, a financial forecasting benchmark spanning FX,
commodities, crypto-assets, fixed income, equities, ETFs, and macroeconomic
indicators. Models are ranked per (tier, spec, metric) cell; *Avg. Rank* is the
geometric mean of per-cell ranks and *Rank Sum* is the sum of the three per-tier
placements (lower is better). 44 models were scored on identical windows.

| # | Model | Tier 1 (Point) | Avg. Rank | Tier 2 (IC) | Avg. Rank | Tier 3 (Portfolio) | Avg. Rank | Rank Sum |
|---|---|---|---|---|---|---|---|---|
| **1** | **EXAONE Forecast for Finance** | **1** | **6.53** | **1** | **7.45** | **1** | **7.43** | **3** |
| 2 | Chronos-2 (Synthetic) | 9 | 11.68 | 3 | 8.74 | 2 | 9.90 | 14 |
| 3 | Reverso (Small) | 6 | 9.60 | 2 | 8.64 | 6 | 13.04 | 14 |
| 4 | TiRex-1.1 | 2 | 6.98 | 4 | 9.71 | 14 | 15.76 | 20 |
| 5 | Chronos-2 | 8 | 10.14 | 13 | 16.12 | 9 | 15.17 | 30 |

The three tiers score complementary objectives: **Tier 1** point accuracy (MASE,
hit rate), **Tier 2** cross-sectional ranking skill (information coefficient), and
**Tier 3** realized portfolio performance (return, Sharpe, volatility, maximum
drawdown). EXAONE Forecast for Finance ranks first in all three for a perfect rank sum of 3,
ahead of the strongest baseline at 14, and is the only model that wins its
head-to-head comparison against all 43 baselines (per-opponent win rate
0.51–0.90). At 202M parameters it sits on the Pareto frontier, outperforming
models more than an order of magnitude larger. Full protocol and results are in
the technical report.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (a CUDA build is recommended; CPU works for small workloads)

## Quickstart

```bash
pip install "exaone-forecast[finance] @ git+https://github.com/LGAI-Research/EXAONE-Forecast.git"
```

```python
import numpy as np
from exaone_forecast.finance import from_pretrained

fc = from_pretrained(device="cuda:0")     # downloads these weights on first use

series = [np.random.randn(500).cumsum() + 100 for _ in range(4)]

q = fc.predict(series, horizon=20)                       # (4, 21, 20) quantiles
yhat = fc.point(series, horizon=20)                      # (4, 20) median forecast
lo, hi = fc.interval(series, horizon=20, lower=0.1, upper=0.9)
```

Inputs may be a single 1D series, a list of 1D arrays of differing lengths, or a
2D `(n_series, T)` array. Feed values on their natural scale — the model
normalizes internally and reads only the last `context_length` (512) points.
Missing observations are supported: encode them as `np.nan`.

## Available checkpoints

| Name | File | Parameters |
|---|---|---|
| `default` | `exaone-finance-1.0.safetensors` | 202M |

`config.json` accompanies the weights and describes the architecture.

## Intended use

Zero-shot probabilistic forecasting of financial and other real-valued time
series, for research and educational purposes. Each series is forecast from its
own recent history; no per-dataset training is required.

## Limitations

- Results are reported on a financial benchmark; broad general-domain
  benchmarking is future work.
- Only zero-shot performance is reported. Fine-tuning and ensembling are untested.
- In this release every series is treated as an individual channel. The
  group-aware mixer supports richer cross-series grouping, which is not exercised
  by this checkpoint.
- Forecasts are statistical outputs, not financial advice. Do not use them as the
  sole basis for an investment decision.

## License

The model weights in this repository are released under the **EXAONE AI Model
License Agreement 1.2 - NC** ([`LICENSE`](./LICENSE)), which limits use to
non-commercial research and education.

The inference code at
[LGAI-Research/EXAONE-Forecast](https://github.com/LGAI-Research/EXAONE-Forecast)
is licensed separately under the BSD-3-Clause-LG AI Research License, which
permits commercial use. Installing that package does not grant commercial rights
to the weights it downloads.

Third-party open source components and their licenses are listed in
[Notice.md](./Notice.md).

## Citation

```bibtex
@article{lgai2026exaonefinance,
  title   = {EXAONE Finance 1.0: An Attention-free Time Series Foundation Model for Financial Time Series},
  author  = {Lee, Seunghan and Lee, Jaehoon and Seo, Jun and Lim, Tae Yoon and
             Kang, Dongwan and Choi, Hwanil and Kim, Minjae and Yoo, Sungdong and
             Kang, Junhyeok and Han, Sangjun and Lee, Soonyoung and Ahn, Wonbin},
  journal = {arXiv preprint arXiv:2609.04239},
  year    = {2026}
}
```

> The code package and its API use the short form **EXAONE Finance**
> (`from exaone_forecast.finance import from_pretrained`).

## Contact

LG AI Research — https://www.lgresearch.ai
