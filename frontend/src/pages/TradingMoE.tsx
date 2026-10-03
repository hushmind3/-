import { api } from '../api';
import { usePolling } from '../hooks';
import { bytes, fmt, num, obj, state, str, time } from '../data';
import { Books, FillTable, ReplayCards } from '../components/operations';
import {
  ActionButton,
  Badge,
  Buttons,
  Card,
  DataPanel,
  ErrorState,
  Loading,
  StatCard,
  Stats,
} from '../components/ui';
export function TradingMoE() {
  const q = usePolling('/api/trading-moe/status', 1000),
    r = q.data;
  if (!r) return q.error ? <ErrorState message={q.error} retry={q.refresh} /> : <Loading />;
  const d = obj(r.decision),
    l = obj(r.learning),
    c = obj(r.compute),
    running = !!r.alive;
  return (
    <>
      {q.error && <ErrorState message={q.error} retry={q.refresh} />}
      <Card
        title="TradingMoE · 자동 가상매매"
        badge={
          <Badge tone={r.error ? 'bad' : running ? 'good' : 'neutral'}>
            {r.gpu_waiting ? 'GPU 차례 대기' : state(r.status)} · PAPER
          </Badge>
        }
      >
        <Buttons>
          <ActionButton
            label="TradingMoE 시작"
            path={api.moe('start')}
            disabled={running || r.status === 'loading'}
          />
          <ActionButton
            label="TradingMoE 저장 후 정지"
            path={api.moe('stop')}
            disabled={!running}
            pendingLabel="저장 · 정지 요청 중…"
          />
        </Buttons>
        {r.error && (
          <p role="alert" className="bad">
            {str(r.error)}
          </p>
        )}
        <p className="muted">
          {str(r.checkpoint)} · PID {str(r.pid)} · 적재 {fmt(r.load_count, 0)}회
        </p>
        {r.gpu_waiting && <p className="notice">다른 모델의 GPU 작업이 끝나면 자동으로 이어 실행합니다.</p>}
      </Card>
      <Stats>
        <StatCard
          title="모델"
          value={fmt(num(r.parameters) / 1e9, 4) + 'B'}
          detail={bytes(r.checkpoint_bytes) + ' · ' + fmt(r.expert_count, 0) + '개 expert'}
        />
        <StatCard
          title="실제 계산 장치"
          value={running ? str(c.inference_device) : '정지 · 계산 없음'}
          badge={
            <Badge tone={running && str(c.inference_device).includes('cuda') ? 'good' : 'neutral'}>
              {str(c.gpu_name)}
            </Badge>
          }
          detail={
            '학습 ' +
            (running ? str(c.learning_device) : '정지') +
            ' · VRAM ' +
            bytes(running ? c.allocated_bytes : 0) +
            ' · Worker RAM ' +
            bytes(running ? r.worker_ram_bytes : 0)
          }
        />
        <StatCard
          title="가중치 업데이트"
          value={fmt(r.optimizer_updates, 0) + '회'}
          badge={
            <Badge tone={running ? 'good' : 'neutral'}>
              {running ? '실행 중' : '최근 저장 기록'}
            </Badge>
          }
          detail={'loss ' + fmt(l.loss, 6) + ' · ' + time(l.updated_at)}
        />
        <StatCard
          title="최근 판단 소요"
          value={fmt(d.seconds, 3) + '초'}
          detail={'전체 사이클 ' + fmt(r.cycle_seconds, 3) + '초 · ' + time(r.updated_at)}
        />
      </Stats>
      <Card
        title="현재 의사결정"
        badge={
          <Badge tone={running ? 'good' : 'neutral'}>
            {running ? '현재 실행' : '마지막 판단 기록'}
          </Badge>
        }
      >
        <strong className="stat">ETHUSDT · {str(d.action)}</strong>
        <p className="decision">
          현재 비중 {fmt(num(d.current_weight) * 100)}% → 목표 {fmt(num(d.target_weight) * 100)}% ·
          현금 목표 {fmt(num(d.cash_weight) * 100)}%
        </p>
        <p className="muted">
          시장 시점 {time(d.as_of || r.market_timestamp)} · Controller value {fmt(d.value, 6)}
        </p>
        <div className="pipeline">
          시장 expert 8개 <span>→</span> Market State <span>→</span> MacroHFT 6개 <span>→</span>{' '}
          Controller <span>→</span> {str(d.action)}
        </div>
        <DataPanel title="실제 계층별 사용 상태" data={r.stages} />
      </Card>
      <Books books={r.books} eth />
      <Stats>
        <StatCard
          title="최근 reward"
          value={fmt(l.reward_points ?? r.reward_points, 6)}
          detail="비용 차감 가상계좌 결과"
        />
        <StatCard title="최근 loss" value={fmt(l.loss, 6)} detail={time(l.updated_at)} />
      </Stats>
      <ReplayCards replay={obj(r.replay)} />
      <Card title="최근 가상체결">
        <FillTable fills={r.fills} eth />
      </Card>
      <DataPanel title="대기 주문 · 계좌 · 학습 · 저장 상세" data={r} />
    </>
  );
}
