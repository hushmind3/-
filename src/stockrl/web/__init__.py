"""웹 기능별 구현. 진입점은 stockrl.web_app입니다.

runtime.py: 서버가 관리하는 feed/agent 실행·중지와 독립 제어
status.py: 대시보드 상태 조회와 측정값 캐시
accounts.py: 사용자가 요청한 가상계좌 초기화
health.py: 시세·관찰 지연과 시장 세션 판정
server.py: HTTP API와 화면 파일 제공
resources.py: 프로젝트 경로와 공개 화면 파일 목록
assets/: HTML에서 읽는 CSS 및 제어·학습·계좌·운영 표시 JS
"""
