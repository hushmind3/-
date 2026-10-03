"""온라인 에이전트 기능별 구현. 실행 진입점은 stockrl.global_online입니다.

observation.py: 실시간 시세 관찰, 두 모델 추론, 가상계좌, 독립 제어
learning.py: 순차 replay 학습, GPU 작업 배분, 학습 완료 가중치 반영
rewards.py: stockrl.rewards의 공용 보상 구현 import
validation.py: 고정 시험본 대결과 승급 판정
evaluation.py: 오프라인 계좌 평가, 백테스트, 추론 시간 측정
checkpoint.py: 모델 저장과 읽기
data.py: 기존 learner 측정 및 stockrl.experience의 호환 import
"""
