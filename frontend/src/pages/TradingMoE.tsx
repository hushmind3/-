import { api } from '../api';
import { usePolling } from '../hooks';
import { bytes, fmt, list, num, obj, pct, str, time } from '../data';
import { FillHistory } from '../components/operations';
import { ActionButton, DataTable, Disclosure, ErrorState, JsonDetails, Loading, Signal } from '../components/ui';

function PipelineMark({ state }: { state: 'done' | 'active' | 'waiting' }) {
  return <span className={'moe-pipeline-mark ' + state}>{state === 'done' ? '완료' : state === 'active' ? '진행 중' : '대기'}</span>;
}

export function TradingMoE() {
  const q = usePolling('/api/trading-moe/status', 1000), r = q.data;
  if (!r) return q.error ? <ErrorState message={q.error} retry={q.refresh} /> : <Loading />;

  const decision = obj(r.decision), learning = obj(r.learning), compute = obj(r.compute), stages = obj(r.stages), replay = obj(r.replay);
  const live = !!r.alive;
  const books = Object.entries(obj(r.books)).map(([currency, value]) => {
    const book = obj(value), pnl = num(book.net_pnl ?? (num(book.equity) - num(book.initial_cash)));
    return [currency, fmt(book.equity) + ' ' + currency, fmt(book.cash), fmt(pnl) + ' · ' + pct(book.net_return_rate ?? (num(book.initial_cash) ? pnl / num(book.initial_cash) : 0)), fmt(book.trade_count, 0) + '회', fmt(book.costs ?? (num(book.fees) + num(book.slippage) + num(book.spread) + num(book.sell_tax)))];
  });
  const positions = Object.entries(obj(r.books)).flatMap(([currency, value]) => {
    const book = obj(value), raw = book.positions;
    const entries = Array.isArray(raw) ? raw.map((position) => [str(obj(position).symbol), position] as const) : Object.entries(obj(raw));
    return entries.map(([symbol, value]) => {
      const position = obj(value), mark = obj(book.marks)[symbol] ?? position.current_price ?? position.price;
      const quantity = num(position.quantity), average = num(position.average_cost), equity = num(book.equity);
      const weight = position.weight ?? (equity > 0 ? quantity * num(mark) / equity : undefined);
      return [symbol, currency, fmt(quantity, 6), fmt(average, 6), fmt(mark, 6), fmt((num(mark) - average) * quantity, 4), pct(weight)];
    });
  });
  const action = str(decision.action, '대기');
  const eligible = num(replay.eligible);
  const positionCount = positions.length;

  return <>
    {q.error && <ErrorState message={q.error} retry={q.refresh} />}

    <section className="moe-operations-bar" aria-label="TradingMoE 실행 제어">
      <div className="moe-operations-status">
        <Signal label="독립 worker" value={r.gpu_waiting ? 'GPU 대기' : str(r.status)} tone={r.error ? 'bad' : live ? 'good' : 'neutral'} />
        <span>{live ? 'PAPER 운용 중' : 'PAPER 정지'}</span>
      </div>
      <div className="action-row">
        <ActionButton label="TradingMoE 시작" path={api.moe('start')} disabled={live || r.status === 'loading'} />
        <ActionButton label="TradingMoE 저장 후 정지" path={api.moe('stop')} disabled={!live} pendingLabel="저장 · 정지 요청 중…" />
      </div>
      {r.error && <p className="error-text">{str(r.error)}</p>}
    </section>

    <section className="moe-decision-hero" aria-label="현재 최종 판단">
      <div className="moe-decision-main">
        <div className="moe-decision-kicker"><span>현재 결정</span><time>{time(decision.as_of || r.market_timestamp)}</time></div>
        <div className="moe-instrument-action">
          <h2>ETHUSDT</h2>
          <strong className={'action-word ' + action}>{action}</strong>
        </div>
        <div className="moe-weight-compare">
          <div><small>현재 비중</small><strong>{pct(decision.current_weight)}</strong></div>
          <span aria-hidden="true">→</span>
          <div><small>목표 비중</small><strong>{pct(decision.target_weight)}</strong></div>
          <div className="moe-cash-target"><small>현금 목표</small><strong>{pct(decision.cash_weight)}</strong></div>
        </div>
      </div>
      <aside className="moe-decision-latency">
        <small>판단 소요</small>
        <strong>{fmt(decision.seconds, 3)}<em>초</em></strong>
        <span>전체 cycle {fmt(r.cycle_seconds, 3)}초</span>
      </aside>
      <div className="moe-system-metadata" aria-label="모델 실행 자원">
        <span><small>계산 장치</small><b>{live ? str(compute.inference_device, '확인 중') : '정지'} · {str(compute.gpu_name, 'GPU 정보 없음')}</b></span>
        <span><small>GPU / RAM</small><b>{bytes(live ? compute.allocated_bytes : 0)} / {bytes(live ? r.worker_ram_bytes : 0)}</b></span>
        <span><small>모델 크기</small><b>{fmt(num(r.parameters) / 1e9, 3)}B · {bytes(r.checkpoint_bytes)}</b></span>
        <span><small>PID · 적재</small><b>{str(r.pid, '—')} · {fmt(r.load_count, 0)}회</b></span>
        <span><small>Optimizer 누계</small><b>{fmt(r.optimizer_updates, 0)}회</b></span>
      </div>
    </section>

    <section className="moe-pipeline-region" aria-label="추론부터 학습까지 처리 단계">
      <header className="moe-section-heading"><div><small>INFERENCE → PAPER → LEARNING</small><h2>처리 흐름</h2></div><span>{live ? 'worker 실행 상태 기준' : '마지막 저장 상태 기준'}</span></header>
      <div className="moe-pipeline">
        <section className="moe-pipeline-group market" aria-label="시장 해석 단계">
          <header><span>01</span><div><small>시장 해석</small><PipelineMark state={stages.market && stages.state ? 'done' : 'waiting'} /></div></header>
          <div className="moe-pipeline-line">
            <div><small>Market experts</small><strong>{fmt(r.expert_count, 0)}개</strong><span>{stages.market ? '입력 분석 완료' : '새 입력 대기'}</span></div>
            <i aria-hidden="true">→</i>
            <div><small>Routing · Fusion</small><strong>{stages.state ? 'Market State 준비' : '출력 대기'}</strong><span>전문가 결과를 공통 상태로 통합</span></div>
          </div>
        </section>
        <section className="moe-pipeline-group decision" aria-label="최종 의사결정 단계">
          <header><span>02</span><div><small>최종 결정</small><PipelineMark state={stages.controller ? 'done' : stages.policy ? 'active' : 'waiting'} /></div></header>
          <div className="moe-pipeline-line">
            <div><small>Policy · Controller</small><strong>{stages.policy ? '정책 반영' : '정책 입력 대기'}</strong><span>{stages.controller ? 'controller 계산 완료' : '최종 결정 대기'}</span></div>
            <i aria-hidden="true">→</i>
            <div className="moe-pipeline-action"><small>Action · 목표 비중</small><strong>{action} · {pct(decision.target_weight)}</strong><span>현재 {pct(decision.current_weight)} → 목표 {pct(decision.target_weight)}</span></div>
          </div>
        </section>
        <section className="moe-pipeline-group learning" aria-label="가상 실행과 학습 단계">
          <header><span>03</span><div><small>가상 실행 · 학습</small><PipelineMark state={eligible ? 'active' : 'waiting'} /></div></header>
          <div className="moe-pipeline-line">
            <div><small>Paper 체결</small><strong>{fmt(list(r.fills).length, 0)}건 누적</strong><span>기존 가상계좌 기록</span></div>
            <i aria-hidden="true">→</i>
            <div><small>Reward · Replay</small><strong>학습 가능 {fmt(eligible, 0)}건</strong><span>최근 reward {fmt(learning.reward_points ?? r.reward_points, 6)}</span></div>
            <i aria-hidden="true">→</i>
            <div><small>Optimizer</small><strong>{fmt(r.optimizer_updates, 0)}회 update</strong><span>최근 loss {fmt(learning.loss, 6)}</span></div>
          </div>
        </section>
      </div>
    </section>

    <section className="moe-portfolio-region">
      <header className="moe-section-heading"><div><small>PAPER PORTFOLIO</small><h2>가상계좌와 보유 포지션</h2></div><span>{positionCount}개 포지션</span></header>
      <div className="moe-portfolio-summary"><DataTable headers={['통화','NAV','현금','누적 손익 · 수익률','체결','총 비용']} rows={books} /></div>
      <div className="moe-position-table"><DataTable headers={['종목','통화','수량','평단','현재가','평가손익','비중']} rows={positions} empty="현재 보유 포지션이 없습니다." /></div>
    </section>

    <div className="moe-support-grid">
      <section className="moe-learning-region">
        <header className="moe-section-heading"><div><small>LEARNING STATUS</small><h2>최근 학습</h2></div><span>{time(learning.updated_at)}</span></header>
        <div className="moe-learning-metrics">
          <div><small>Reward</small><strong>{fmt(learning.reward_points ?? r.reward_points, 6)}</strong></div>
          <div><small>Loss</small><strong>{fmt(learning.loss, 6)}</strong></div>
          <div><small>Samples</small><strong>{fmt(learning.samples, 0)}</strong></div>
          <div><small>학습 시간</small><strong>{fmt(learning.seconds, 3)}초</strong></div>
          <div><small>Replay 전체 / 학습 가능</small><strong>{fmt(replay.total, 0)} / {fmt(replay.eligible, 0)}</strong></div>
          <div><small>결과 대기</small><strong>{fmt(replay.pending, 0)}건</strong></div>
        </div>
      </section>
      <section className="moe-fills-region">
        <header className="moe-section-heading"><div><small>RECENT PAPER FILLS</small><h2>최근 가상 체결</h2></div><span>{list(r.fills).length}건</span></header>
        <FillHistory fills={r.fills} eth />
      </section>
    </div>

    <Disclosure title="시스템 진단 · 체크포인트 경로 · 원본 상태">
      <p className="moe-checkpoint-path">{str(r.checkpoint)}</p>
      <JsonDetails title="최근 계산 장치와 단계 상태" data={{ compute, stages, decision: r.decision, error: r.error }} />
      <JsonDetails title="원본 runtime 상태" data={r} />
    </Disclosure>
  </>;
}
