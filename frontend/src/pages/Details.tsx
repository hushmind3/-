import { usePolling } from '../hooks';
import { str } from '../data';
import type { StatusResponse } from '../types';
import { ResourceCards } from '../components/operations';
import { Card, DataPanel, ErrorState } from '../components/ui';
export function Details({ status: s }: { status: StatusResponse }) {
  const r = usePolling('/api/runtime');
  return (
    <>
      <ResourceCards status={s} />
      {r.error && <ErrorState message={r.error} />}
      <DataPanel title="RAM · GPU 작업 · 프로세스 · 자원 상세" data={r.data} />
      <Card title="최근 서버 로그">
        <div className="log-list">
          {(typeof s.logs === 'string' ? s.logs.split('\n') : s.logs || [])
            .slice(-60)
            .reverse()
            .map((line, i) => (
              <pre key={i}>{str(line)}</pre>
            ))}
        </div>
      </Card>
      <DataPanel title="학습 · 추론 · 저장 시간 · 최근 오류" data={s.metrics} />
      <DataPanel title="시장 수집 · 저장 DB · 입력 기록" data={s.feed_metrics} />
      <DataPanel title="주문 출력 · 행동 분포" data={s.output_diagnostics} />
      <DataPanel title="계좌 대조 · 비용 계산" data={s.account_observability} />
      <DataPanel title="장기 흐름 · 종목 확장" data={s.universe_expansion} />
      <DataPanel title="Backtest · 전체 실행 설정" data={s.backtest} />
    </>
  );
}
