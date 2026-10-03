# Python 코드 정리

## 분석 범위와 호출 경로

정리 전 Git에 등록된 Python 1,021개를 한 번 AST로 조사했다. 프로젝트 코드·스크립트·테스트 119개(소스 61, 스크립트 48, 테스트 9, 시작 파일 1), 원본 vendor 902개다. Vendor 7개는 원래부터 문법이 유효하지 않은 미사용 테스트/노트북 export였다. 원본 expert 소스와 가중치는 변경하지 않았다.

| 실행 경로 | 책임 |
| --- | --- |
| start_stockrl → launch_web → web/server | Python 웹서버, 기존 endpoint, React 정적 제공 |
| web/runtime, web/trading_moe, web/workers | Feed와 모델 독립 lifecycle, 재시작 시 프로세스 재연결 |
| run_native_vertical_trading → trading_moe → moe_native/expert_backends/moe_stock_policies | PT 한 번 적재, 등록된 expert 순차 추론, adapter/router/fusion/controller |
| assembly_orchestrator → run_assembly_trial | 작은 recipe/state 후보 생성·시험·교체, 공유 PT 유지 |
| moe_paper → paper_account / online/rewards / replay_store | 가상 체결·계좌·지연 보상·경험 저장 |
| moe_training | 현재 controller/adapter optimizer 업데이트 |
| global_online / online/* | 현재 제공되는 기존 모델 lifecycle·API 호환 경로 |
| cli/core/public_teachers/research_ingest, scripts/research | 명시적 CLI·선택적 연구 도구 |

참조 여부는 import/call/attribute와 문자열·CLI dispatch를 함께 확인했다. 정적 미참조만으로 vendor나 PyTorch forward/load 훅을 제거하지 않았다. 상세 1회 분석 결과와 확정 목록은 로컬 `runtime/code-cleanup/source-map.json`, `plan.json`에 있다.

## 확정 목록대로 적용한 변경

- 미사용 `continuous.py` 제거. CLI continuous/replay는 이미 `legacy_checkpoint_command`로 연결되어 있었으며 dispatch는 유지했다.
- 미참조 `MockBroker`, `small_config`, `finrl_imitation_records`, `write_source_manifest` 제거. 사용되는 broker 인터페이스와 연구 명령은 유지했다.
- Native runner AST 변환·컴파일을 최대 8개 코드 객체로 재사용. expert 모델·입력·weight는 이 캐시에 넣지 않는다. call별 namespace와 checkpoint revision source를 유지한다.
- 같은 vendor source를 `sys.modules`에서 재사용. 경로/mtime/크기가 바뀌면 다시 읽는다. Toto 진단/registered 경로의 중복 로더를 통합했다.
- JSON 읽기를 state_io 하나로 통합. worker handoff도 기존 atomic writer를 사용한다. 실패 기본값은 호출자가 독립적으로 소유한다.
- atomic writer의 경로별 잠금은 작업이 끝나면 해제될 수 있게 한다. 저장 파일 이름이 늘어날 때 잠금 객체가 영구 축적되지 않는다.
- paper submission의 동일 features/mask/IDs를 종목마다 복사하지 않고 판단당 한 번 소유한다. 미래 panel 변경과 분리되며 종목별 action/reward/계좌 기록은 따로 유지한다. 전달된 배열은 보상 처리에서 읽기 전용으로 취급한다.

## 보존 범위

모델 형식·checkpoint migration·CPU/GPU residency·최대 GPU expert 1개·현재 endpoint·Champion/Candidate/assembly 상태·계좌·수수료·reward·replay 스키마를 유지했다. 새 거래 기능이나 자동화 상태 파일은 추가하지 않았다. 원본 파일은 프로젝트 밖 `<프로젝트 이름>-휴지통/Python정리-*/`에 보존하고 이동목록을 남겼다.

## 검증

신규 회귀 항목 8개: 코드 재사용/입력·모델 분리, load-only/runner revision, 캐시 상한, source 변경 감지, Toto 로더, JSON 기본값 독립성, concurrent atomic write/잠금 해제, 다종목 입력 공유/체결/보상/replay/재개.

전체 Python 테스트는 수정 완료 후 `python -m unittest discover -s tests -v` 한 번 실행했다. **161개 모두 통과, 14.385초**, 새 회귀 항목 8개 포함. 결과는 로컬 `runtime/code-cleanup/tests.log`에 보존한다. 실행 데이터나 모델을 Git에 포함하지 않는다.
