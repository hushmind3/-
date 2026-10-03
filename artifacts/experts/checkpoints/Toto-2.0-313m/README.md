---
license: apache-2.0
pipeline_tag: time-series-forecasting
library_name: pytorch
tags:
- time-series-forecasting
- foundation-models
- pretrained-models
- time-series
- timeseries
- forecasting
- observability
- safetensors
- pytorch_model_hub_mixin
thumbnail: https://web-assets.dd-static.net/42588/1778691695-toto-2-hero.png
model-index:
- name: Toto-2.0-313m
  results:
  - task:
      type: time-series-forecasting
    dataset:
      name: BOOM
      type: BOOM
    metrics:
    - type: CRPS
      value: 0.351
      name: CRPS
    - type: MASE
      value: 0.585
      name: MASE
    source:
      url: https://huggingface.co/spaces/Datadog/BOOM
      name: BOOM 💥 Observability Time-Series Forecasting Leaderboard
  - task:
      type: time-series-forecasting
    dataset:
      name: GIFT-Eval
      type: GIFT-Eval
    metrics:
    - type: CRPS
      value: 0.481
      name: CRPS
    - type: MASE
      value: 0.703
      name: MASE
    source:
      url: https://huggingface.co/spaces/Salesforce/GIFT-Eval
      name: GIFT-Eval Time Series Forecasting Leaderboard
  - task:
      type: time-series-forecasting
    dataset:
      name: TIME
      type: TIME
    metrics:
    - type: CRPS
      value: 0.535
      name: CRPS
    - type: MASE
      value: 0.642
      name: MASE
    source:
      url: https://huggingface.co/spaces/Real-TSF/TIME-leaderboard
      name: TIME Benchmark Leaderboard
---

# Toto-2.0-313m

Toto (Time Series Optimized Transformer for [Observability](https://www.datadoghq.com/knowledge-center/observability/)) is a family of time series foundation models for multivariate forecasting developed by [Datadog](https://www.datadoghq.com/). Toto 2.0 is the current generation, featuring u-μP-scaled transformers ranging from 4m to 2.5B parameters, all trained from a single recipe. Forecast quality improves reliably with parameter count across the family.

The family sets a new state of the art on three forecasting benchmarks: [BOOM](https://huggingface.co/spaces/Datadog/BOOM), our observability benchmark; [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), the standard general-purpose benchmark; and the recent contamination-resistant [TIME](https://arxiv.org/abs/2602.12147) benchmark.

## 📊 Performance

<figure>
<img src="assets/pareto.png" alt="Pareto frontier on BOOM and GIFT-Eval">
<figcaption>Every Toto 2.0 size sits on or near the Pareto frontier on both BOOM and GIFT-Eval. The three largest sizes rank first, second, and third among foundation models on GIFT-Eval CRPS rank. On TIME, Toto 2.0 sizes take the top three spots on every metric, ahead of every other external foundation model evaluated.</figcaption>
</figure>

## ⚡ Quick Start

Inference code is available on [GitHub](https://github.com/DataDog/toto).

### Installation

```bash
pip install toto-models
```

### Inference Example

```python
import torch
from toto2 import Toto2Model

model = Toto2Model.from_pretrained("Datadog/Toto-2.0-313m")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# (batch, n_variates, time_steps)
target = torch.randn(1, 1, 512, device=device)
target_mask = torch.ones_like(target, dtype=torch.bool)
series_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

# Returns quantiles of shape (9, batch, n_variates, horizon)
# Quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
quantiles = model.forecast(
    {"target": target, "target_mask": target_mask, "series_ids": series_ids},
    horizon=96,
    decode_block_size=768,
    has_missing_values=False,
)
```

For more examples, see the [Quick Start notebook](https://github.com/DataDog/toto/blob/main/toto2/notebooks/quick_start.ipynb) and [GluonTS integration notebook](https://github.com/DataDog/toto/blob/main/toto2/notebooks/gluonts_integration.ipynb).

## 💾 Available Checkpoints

All five Toto 2.0 sizes share the same training recipe; pick a size based on your accuracy/latency budget. Latency is forward-pass time for a 1,024-step single-pass forecast at batch size 8 on a single A100.

| Model | Params | Weights (fp32) | Latency | Recommended for |
|:---:|:---:|:---:|---|---|
| [Toto‑2.0‑4m](https://huggingface.co/Datadog/Toto-2.0-4m)     | 4m   | 16 MB  | ~3.8&nbsp;ms  | Edge / CPU deployment; tightest latency or memory budgets. |
| [Toto‑2.0‑22m](https://huggingface.co/Datadog/Toto-2.0-22m)   | 22m  | 84 MB  | ~5.0&nbsp;ms  | Efficient default — matches or beats Toto 1.0 quality with ~7× fewer parameters. |
| [Toto‑2.0‑313m](https://huggingface.co/Datadog/Toto-2.0-313m) | 313m | 1.2 GB | ~15.4&nbsp;ms | Strong general-purpose checkpoint; top-3 foundation model on GIFT-Eval. |
| [Toto‑2.0‑1B](https://huggingface.co/Datadog/Toto-2.0-1B)     | 1B   | 3.9 GB | ~20.9&nbsp;ms | Best quality / cost tradeoff for production workloads. |
| [Toto‑2.0‑2.5B](https://huggingface.co/Datadog/Toto-2.0-2.5B) | 2.5B | 9.1 GB | ~36.2&nbsp;ms | Highest accuracy; #1 foundation model on every benchmark. |

## ✨ Key Features

- **Zero-Shot Forecasting:** Forecast without fine-tuning on your specific time series.
- **Multi-Variate Support:** Efficiently process multiple variables using alternating time/variate attention.
- **Probabilistic Predictions:** Generate point forecasts and uncertainty estimates via a quantile output head.
- **Decoder-Only Architecture:** Support for variable prediction horizons and context lengths.
- **u-μP Scaling:** A single training recipe transfers cleanly across all five sizes (4m → 2.5B).

## 🏗️ Architecture

<figure>
<img src="assets/architecture.png" alt="Overview of the Toto 2.0 architecture.">
<figcaption>A decoder-only patched transformer whose attention layers alternate between time-axis (causal) and variate-axis (full) views of the input. Toto 2.0 adds <b>contiguous patch masking (CPM)</b> for single-pass parallel decoding, a <b>quantile output head</b> trained with pinball loss, a robust arcsinh input scaler, residual MLP patch projections, and is trained with NorMuon. See the <a href="https://arxiv.org/abs/2605.20119">technical report</a> for details.</figcaption>
</figure>

## 🔗 Additional Resources

- [Technical Report](https://arxiv.org/abs/2605.20119)
- [Blog Post](https://www.datadoghq.com/blog/ai/toto-2/)
- [GitHub Repository](https://github.com/DataDog/toto)
- [Toto 2.0 Collection](https://huggingface.co/collections/Datadog/toto-20) — all five base checkpoints
- [BOOM Dataset](https://huggingface.co/datasets/Datadog/BOOM) — Datadog's observability time-series benchmark
- [Toto 1.0 Weights](https://huggingface.co/Datadog/Toto-Open-Base-1.0)

## 📖 Citation

```bibtex
@misc{khwaja2026toto20timeseries,
      title={Toto 2.0: Time Series Forecasting Enters the Scaling Era}, 
      author={Emaad Khwaja and Chris Lettieri and Gerald Woo and Eden Belouadah and Marc Cenac and Guillaume Jarry and Enguerrand Paquin and Xunyi Zhao and Viktoriya Zhukov and Othmane Abou-Amal and Chenghao Liu and Ameet Talwalkar and David Asker},
      year={2026},
      eprint={2605.20119},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.20119}, 
}
```