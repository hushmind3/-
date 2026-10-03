# 다른 Windows 컴퓨터에 설치

1. Python 3.13과 CUDA 13.2를 지원하는 NVIDIA 드라이버를 설치한다.
2. GitHub 프로젝트를 다운로드하거나 clone한다. `설치.cmd`를 실행한다.
3. 기존 바탕화면 `모델` 폴더를 새 컴퓨터의 바탕화면으로 복사한다. 모델 가중치는 GitHub에 포함하지 않는다.
4. `서버켜기.cmd`를 실행한다. 웹서버만 시작하며 모델은 화면의 시작 버튼으로 켠다.

설치 프로그램은 프로젝트 위치와 새 사용자 홈을 기준으로 경로를 생성한다. 사용자 이름이나 예전 컴퓨터의 절대 경로를 입력할 필요가 없다. 원본 모델 파일·가상계좌를 초기화하거나 자동 거래를 시작하지 않는다.

## 전달되는 설정

- `configs/web_settings.default.json`: 처음 실행할 때 사용할 모드 설정. 모델 시작은 기본 OFF.
- `configs/online_learning.json`: optimizer·replay·보상 관련 운영 설정.
- `src/stockrl/replay_store.py`: SQLite 스키마와 migration. 새 replay 사용 시 자동 생성.
- `src/stockrl/paper_account.py`, `moe_paper.py`: 새 가상계좌와 별도 MoE replay 생성.
- `artifacts/experts/registry.template.json`: 20개 expert의 경로 독립 메타데이터. 첫 사용 시 runtime registry를 생성.
- 공식 ETHUSDT `df_val.feather`: 현재 과거 가상매매 경로에 필요한 원본 입력. DB/replay와 구분해서 프로젝트에 포함.

## 전달하지 않는 실행 데이터

`runtime`, 컴퓨터별 `configs/local`, Python 환경, replay·DB·로그·프로세스 PID·캐시, 모델 가중치, 휴지통은 GitHub에서 제외한다. 새 컴퓨터에서는 빈 실행 상태로 시작한다. 누적 계좌·경험을 이어가려면 모델을 정상 정지한 뒤 별도로 runtime을 복사한다.

현재 worker는 공식 ETHUSDT 과거 구간을 사용한다. Feed 수집 종목 수를 실시간 MoE 매매 종목 수로 해석하지 않는다. 새로운 컴퓨터에서 전체 CUDA 설치와 expert 추론은 해당 장치에서 확인해야 한다.
