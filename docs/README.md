# 문서 찾기

운영 안내와 측정 결과를 구분합니다. 현재 프로젝트의 시작점은 [루트 README](../README.md)입니다.

## 운영

- [새 Windows 컴퓨터 설치](windows-install.md)
- [폴더·모델·실행 상태 저장 위치](project-storage.md)
- [React 화면과 API 기능 대조](frontend.md)
- [Champion/Candidate 실행 제어](model-lifecycle.md)
- [TradingMoE 실행·GPU·학습](trading-moe-learning.md)
- [자동조립·후보 생성·평가](assembly.md)
- [기존 Transformer의 Mac 실행 안내](mac-install.md) — TradingMoE CUDA 실행과 구분

## 모델 구조

- [주식 매매 정책 전문가](stock_policy_experts.md)
- [전문가별 통합 설계 조사](heterogeneous-experts-design.md)

## 결과 기록

`reports/`에는 특정 실행 당시의 측정과 결과를 보존합니다. 현재 실행 상태나 현재 설정값은 대시보드/API로 확인합니다.

- [초기 wrapper 실행 구조](reports/expert-wrapper-runtime.md)
- [초기 wrapper 추론·측정 범위](reports/expert-wrapper-inference.md)
- [원본 전문가별 측정](reports/frozen-expert-measurements.md)
- [통합 추론 시간 측정](reports/trading-moe-inference-measurements.md)
- 원본 JSON: `reports/trading-moe-artifacts.json`, `trading-moe-execution-result.json`, `trading-moe-lifecycle-result.json`
- 과거 정리 기록: `reports/server-cleanup.md`, `reports/moe-runtime-cleanup.md`

- [Python 코드 정리 · 호출 경로·보존 범위·검증](reports/python-code-cleanup.md)

- [TradingMoE 공용 부품 분리](reports/shared-infrastructure.md)
