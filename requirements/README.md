# Python 의존성

| 파일 | 사용 |
| --- | --- |
| `base.txt` | 기본 프로젝트·공개 정책 adapter |
| `moe.txt` | Windows CUDA TradingMoE 기본 expert 환경 |
| `moe-toto.txt` | 별도 Toto 환경. 다른 transformers 버전 |
| `mac.txt` | 기존 Mac Transformer/MPS 환경 |

`설치.cmd` → `scripts/install_windows.py`가 `moe.txt`와 `moe-toto.txt`를 각각 설치합니다. 두 목록은 같은 폴더의 `base.txt`를 포함합니다. 현재 실행 중인 환경을 이 정리 작업에서 재설치하지 않았습니다.

React 의존성은 `frontend/package.json`과 `package-lock.json`에 별도로 있습니다.
