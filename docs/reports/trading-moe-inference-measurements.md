# 14개 expert · 통합 추론 실제 측정

Run: `0fc0418b90b94deeb876395cd84e286a` · 입력 기준: 2025-10-31

단위: 초. 매번 새 worker/모델 적재. OS 파일 cache는 따뜻할 수 있습니다.

| Expert | 첫 적재 | GPU 전송 | Forward | 전체 왕복 |
|---|---:|---:|---:|---:|
| FinText TimesFM Global | 3.089 | 0.018 | 0.325 | 4.036 |
| Kronos Base + Tokenizer | 5.213 | 0.100 | 0.343 | 6.576 |
| Toto 2.0 | 6.337 | 0.241 | 0.381 | 8.472 |
| MacroHFT slope/1 | 3.704 | 0.002 | 0.114 | 4.399 |
| MacroHFT slope/2 | 3.748 | 0.002 | 0.112 | 4.448 |
| MacroHFT slope/3 | 3.728 | 0.001 | 0.106 | 4.432 |
| MacroHFT vol/1 | 3.830 | 0.003 | 0.122 | 4.580 |
| MacroHFT vol/2 | 3.876 | 0.001 | 0.123 | 4.576 |
| MacroHFT vol/3 | 3.758 | 0.001 | 0.120 | 4.434 |
| MarketGPT | 4.351 | 0.063 | 0.155 | 9.221 |
| EXAONE Finance | 9.357 | 0.154 | 0.258 | 11.150 |
| FinText Chronos Global | 9.236 | 0.106 | 0.379 | 10.774 |
| Time-MoE Large | 12.607 | 0.250 | 0.388 | 14.679 |
| FinCast 1.0B | 11.703 | 0.848 | 0.882 | 17.015 |

전체 pipeline: **112.948초** · adapter: 0.002초 · CPU fusion: 4.060초

GPU 동시 expert 최대 **1개** · residency 871회 기록 · 완료 후 모든 expert 미적재.

원본 JSON SHA256 및 내용 일치 14개 확인. Fusion/head는 미학습 frozen 진단용, 학습/optimizer/계좌 실행 없음.

실제 일봉 주식 입력과 합성 ITCH/ETH schema fixture를 함께 사용한 경로 검증입니다. 거래 성과 검증이 아닙니다.

[측정 범위·구현 설명](expert-wrapper-inference.md) · [기계 판독 원본](trading-moe-inference-measurements.json)
