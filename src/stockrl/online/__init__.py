"""온라인 에이전트 기능별 구현. 실행 진입점은 stockrl.global_online입니다.

observation.py: 실시간 시세 관찰, 두 모델 추론, 가상계좌, 독립 제어
learning.py: 순차 replay 학습, GPU 작업 배분, 학습 완료 가중치 반영
rewards.py: 실제 가상체결 손익과 과거 행동의 보상 연결
validation.py: 고정 시험본 대결과 승급 판정
evaluation.py: 오프라인 계좌 평가, 백테스트, 추론 시간 측정
checkpoint.py: 모델 저장과 읽기
data.py: 경험 형식, 공통 관찰본, CSV 증분 읽기
"""
