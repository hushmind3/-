# 구형 0.5B 퇴역 · TradingMoE Champion/Candidate 유지

- 구형 GlobalMarketTransformer/OnlineGlobalAgent, 학습·승급·prefix cache·전용 연구 도구와 PySide 구형 화면을 외부 휴지통으로 이동했습니다. 모델 원본 PT는 변경하지 않았습니다.
- Champion/Candidate는 역할 이름으로 유지합니다. 두 운영 모델의 시작/정지는 각 모델 worker에 연결합니다. 조립 시험은 별도의 시험 버튼으로 실행하며 운영계좌 표시를 대체하지 않습니다. 후보는 공용 Champion PT와 작은 recipe/학습 state를 사용하며 전체 PT를 복사하지 않습니다. Candidate 운영 버튼이 조립 시험을 시작하던 연결은 `0df48a1`에서 수정했습니다.
- 시장 Feed는 모델과 독립입니다. 중복 시작을 방지하고 시작 실패 시 요청 상태를 복구합니다. 실패한 모델을 자동 재적재하는 루프도 없앴습니다.
- 웹서버가 모델 종류 확인을 위해 torch.load()하던 경로를 제거했습니다. 모델 적재는 해당 worker에서만 수행합니다.
- 상태 API는 현재 MoE worker/가상계좌에서 계산합니다. 구형 metrics·관찰 큐·승급 기록으로 현재 실행이나 GPU·학습량을 표시하지 않습니다. /api/runtime도 같은 상태를 사용합니다.
- 수동 장기계좌 초기화는 MoE 장기 ledger만 대상으로 합니다. 조립 시험 ledger, recipe, trainable state는 보존합니다. 자동 일별 장기계좌 초기화는 제거했습니다.
- 기존 /#promotionTrial 주소는 같은 MoE 조립 평가 컴포넌트를 사용합니다. 구형 Transformer 시험 UI를 복제 유지하지 않습니다.
- 시장 패널, Experience, 보상, paper account, replay, GPU scheduler와 pretrained expert는 보존했습니다. online/data.py와 online/rewards.py는 저장된 경험/import의 작은 호환 alias만 남았으며 구형 모델 실행기는 없습니다.
- 혼합 테스트 파일에서 공용 데이터·계좌·replay·GPU 검사를 추출하고, MoE lifecycle 검사 10개를 추가했습니다.

## PNG 정리

experts-desktop.png / experts-mobile.png / experts-raw-output.png 총 311,883바이트를 외부 휴지통으로 이동했습니다. 이를 쓰던 browser_experts.cjs는 구형 vanilla DOM/14개 expert를 고정한 옛 검사여서 함께 퇴역했습니다. verification/*.png는 Git에서 제외하며 JSON 원본 출력·검증 결과는 유지합니다. 이번 브라우저 확인은 스크린샷 파일을 생성하지 않았습니다.

## 검증

- 전체 Python suite 1회: 118개 실행, 최초 113개 통과/5개 오류. 테스트 fixture 누락 3개, 제거한 health alias의 웹 진입점 잔재 1개, 빌드와 동시 실행에 따른 stale asset 목록 1개를 해결했습니다. 실패 5개와 asset entrypoint를 대상으로 재실행한 6개 모두 통과했습니다. 전체 suite를 반복하지 않았습니다.
- TypeScript typecheck/Vite build 성공. 프론트 테스트 37개 통과.
- 실제 Python API /status, /runtime, /assembly 정상. 실제 브라우저 두 MoE 카드와 승급전→조립 평가 표시 확인, 브라우저 콘솔 오류 없음. 서버 재시작은 웹만 수행했습니다.
- 무거운 expert/PT 재검증이나 학습 실험은 이번 코드 퇴역 작업에서 실행하지 않았습니다.
