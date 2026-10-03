import { memo } from 'react';
import { api, endpoints } from '../api';
import { at, bytes, fmt, num, obj, pct, rows, state, str, time } from '../data';
import type { Data, Json, ModelRuntime, Role, StatusResponse } from '../types';
import {
  ActionButton,
  Badge,
  Buttons,
  Card,
  DataPanel,
  Display,
  Panel,
  StatCard,
  Stats,
  Table,
  Toggle,
} from './ui';

export function ModeControls({ status: s }: { status: StatusResponse }) {
  return (
    <Card title="실행할 작업 선택">
      <Buttons>
        <Toggle
          label="모델 판단"
          enabled={!!s.observe_enabled}
          path={endpoints.modes}
          field="observe_enabled"
        />
        <Toggle
          label="가상 체결"
          enabled={!!s.paper_enabled}
          path={endpoints.modes}
          field="paper_enabled"
        />
        <Toggle
          label="경험 학습"
          enabled={!!s.learning_enabled}
          path={endpoints.modes}
          field="learning_enabled"
        />
      </Buttons>
      <p className="muted">
        판단은 행동 결정 · 체결은 가상계좌 변경 · 학습은 저장된 경험으로 가중치 업데이트
      </p>
      <DataPanel
        title="각 작업의 실행 범위 · 실제 적용 설정"
        data={{
          observe_enabled: s.observe_enabled,
          paper_enabled: s.paper_enabled,
          learning_enabled: s.learning_enabled,
          autonomy_enabled: s.autonomy_enabled,
          rules: at(s, 'metrics.operating_rules'),
        }}
      />
    </Card>
  );
}
export function SystemControls({ status: s }: { status: StatusResponse }) {
  return (
    <Card
      title="시스템 명령"
      badge={
        <Badge tone={s.real_orders_enabled ? 'bad' : 'neutral'}>
          실제 주문 {s.real_orders_enabled ? 'ON' : 'OFF'}
        </Badge>
      }
    >
      <Buttons>
        <ActionButton
          label={s.feed_running ? '시장 Feed 실행 중' : '시장 Feed 시작'}
          disabled={!!s.feed_running}
          path={endpoints.start}
          body={{ mode: 'live' }}
        />
        <ActionButton label="시세 재연결" path={endpoints.reconnect} />
        <ActionButton label="웹서버만 재시작" path={endpoints.serverRestart} />
        <ActionButton label="전체 정지" path={endpoints.stop} tone="bad" />
      </Buttons>
      <Buttons>
        <ActionButton
          label="두 장기 운영계좌 초기화"
          path={endpoints.resetAccounts}
          tone="bad"
          confirm="Champion과 Candidate의 장기 가상계좌를 초기화합니다. 모델 가중치와 replay는 유지합니다. 계속할까요?"
        />
      </Buttons>
      <p className="muted">시스템 시작은 Feed만 실행합니다. 모델은 각각 시작하세요.</p>
    </Card>
  );
}

export function FillTable({ fills, eth = false }: { fills: Json | undefined; eth?: boolean }) {
  return (
    <Table
      headers={['시각', '종목', '행동', '수량', '체결가', '수수료', '실현손익']}
      rows={rows(fills)
        .slice(-30)
        .reverse()
        .map((f) => [
          time(f.timestamp || f.date || f.time),
          str(f.symbol),
          str(f.side || f.action),
          fmt(num(f.quantity) * (eth ? 0.001 : 1), 6),
          fmt(num(f.price || f.fill_price) * (eth ? 1000 : 1), 4),
          fmt(f.fee || f.fees, 4),
          fmt(f.realized_pnl || f.net_pnl, 4),
        ])}
    />
  );
}
function positions(book: Data): Data[] {
  return Array.isArray(book.positions)
    ? rows(book.positions)
    : Object.entries(obj(book.positions)).map(([symbol, p]) => ({
        symbol,
        ...obj(p),
        mark: obj(book.marks)[symbol],
      }));
}
export const BookCard = memo(
  ({ currency, book, eth = false }: { currency: string; book: Data; eth?: boolean }) => {
    const nav = num(book.equity),
      seed = num(book.initial_cash),
      pnl = book.net_pnl ?? nav - seed,
      rate = book.net_return_rate ?? (seed ? num(pnl) / seed : 0),
      ps = positions(book);
    return (
      <Card title={currency + ' · 가상계좌'} badge={<Badge>PAPER</Badge>}>
        <strong className="stat">
          {fmt(book.equity)} {currency}
        </strong>
        <p className={num(pnl) < 0 ? 'bad' : 'good'}>
          {fmt(pnl)} · {pct(rate)}
        </p>
        <dl className="key-values">
          <div>
            <dt>현금</dt>
            <dd>{fmt(book.cash)}</dd>
          </div>
          <div>
            <dt>시작 자금</dt>
            <dd>{fmt(book.initial_cash)}</dd>
          </div>
          <div>
            <dt>비용 합계</dt>
            <dd>
              {fmt(
                book.costs ??
                  num(book.fees) + num(book.slippage) + num(book.spread) + num(book.sell_tax),
              )}
            </dd>
          </div>
          <div>
            <dt>포지션 / 체결</dt>
            <dd>
              {ps.length}종목 / {fmt(book.trade_count, 0)}회
            </dd>
          </div>
        </dl>
        <Panel title="보유 종목 · 평가손익">
          <Table
            headers={['종목', '수량', '평단', '현재가', '평가손익', '비중']}
            rows={ps.map((p) => {
              const qty = num(p.quantity),
                mark = num(p.mark || p.current_price || p.price),
                avg = num(p.average_cost);
              return [
                str(p.symbol),
                fmt(qty * (eth ? 0.001 : 1), 6),
                fmt(avg * (eth ? 1000 : 1), 4),
                fmt(mark * (eth ? 1000 : 1), 4),
                fmt(p.unrealized_pnl ?? qty * (mark - avg)),
                pct(p.weight ?? (nav ? (qty * mark) / nav : 0)),
              ];
            })}
          />
        </Panel>
        <DataPanel title="수수료 · 세금 · 손익 상세" data={book} />
      </Card>
    );
  },
);
export function Books({ books, eth = false }: { books: Json | undefined; eth?: boolean }) {
  return (
    <div className="two-column">
      {Object.entries(obj(books)).map(([currency, book]) => (
        <BookCard key={currency} currency={currency} book={obj(book)} eth={eth} />
      ))}
    </div>
  );
}
export const ModelCard = memo(
  ({ role, runtime, books }: { role: Role; runtime: ModelRuntime; books: Json | undefined }) => {
    const name = role === 'champion' ? 'Champion' : 'Candidate',
      active = !!runtime.loaded && ['running', 'observing'].includes(str(runtime.status));
    return (
      <Card
        title={name + " · TradingMoE"}
        badge={
          <Badge tone={runtime.error ? 'bad' : active ? 'good' : 'neutral'}>
            {state(runtime.status)}
          </Badge>
        }
      >
        <Buttons>
          <ActionButton
            label={name + ' 시작'}
            path={api.model(role, 'start')}
            disabled={
              runtime.status === 'saving' ||
              ((!!runtime.loaded || !!runtime.requested) && runtime.status !== 'error')
            }
          />
          <ActionButton
            label={name + ' 저장 후 정지'}
            path={api.model(role, 'stop')}
            disabled={runtime.status === 'saving' || (!runtime.loaded && !runtime.requested)}
          />
        </Buttons>
        {runtime.error && (
          <p role="alert" className="bad">
            {str(runtime.error)}
          </p>
        )}
        <dl className="key-values">
          <div>
            <dt>실제 적재 / 계산 장치</dt>
            <dd>
              {runtime.loaded ? '적재됨' : '미적재'} ·{' '}
              {runtime.loaded ? str(runtime.compute_device || runtime.device) : '미실행'}
            </dd>
          </div>
          <div>
            <dt>모델 RAM / VRAM</dt>
            <dd>
              {bytes(runtime.loaded ? runtime.ram_weight_bytes : 0)} /{' '}
              {bytes(runtime.loaded ? runtime.gpu_weight_bytes : 0)}
            </dd>
          </div>
          <div>
            <dt>최근 판단 / 소요 시간</dt>
            <dd>
              {time(runtime.last_decision)} · {fmt(runtime.decision_seconds, 3)}초
            </dd>
          </div>
        </dl>
        <p className="muted">{runtime.account_scope === "trial" ? "조립 시험계좌 · 장기 운영계좌 유지" : "장기 운영계좌"}</p>
        <Books books={books} eth={runtime.source_kind === 'historical_paper'} />
        <DataPanel title={name + ' 실행 · 학습 상세'} data={runtime} />
      </Card>
    );
  },
);
export function ModelCards({ status: s }: { status: StatusResponse }) {
  return (
    <div className="two-column">
      {(['champion', 'candidate'] as const).map((role) => {
        const runtime = s.model_runtime?.[role] || {};
        return (
          <ModelCard
            key={role}
            role={role}
            runtime={runtime}
            books={
              runtime.books ||
              (role === 'champion' ? s.paper_financials : at(s, 'candidate_live_account.books'))
            }
          />
        );
      })}
    </div>
  );
}
export function LearningBoard({ status: s }: { status: StatusResponse }) {
  return (
    <Stats>
      {(['champion', 'candidate'] as const).map((role) => {
        const r = s.model_runtime?.[role] || {},
          l = obj(r.learning),
          live = !!r.loaded && !!r.learning_active;
        return (
          <StatCard
            key={role}
            title={role === 'champion' ? 'Champion · 학습' : 'Candidate · 학습'}
            value={fmt(r.optimizer_updates, 0) + '회 업데이트'}
            badge={
              <Badge tone={live ? 'good' : 'neutral'}>
                {live ? '학습 중' : r.loaded && s.learning_enabled ? '다음 경험 대기' : '학습 중지'}
              </Badge>
            }
            detail={
              <>
                최근 loss {fmt(l.loss, 6)} · reward {fmt(l.reward_points, 6)}
                <br />
                {time(l.updated_at || l.last_update_utc)} · {live ? '현재 실행' : '최근 저장 기록'}
              </>
            }
          />
        );
      })}
    </Stats>
  );
}
export function ReplayCards({ replay: r }: { replay: Data }) {
  const completed =
    r.completed ??
    (Array.isArray(r.daily) ? rows(r.daily).reduce((n, d) => n + num(d.completed), 0) : undefined);
  return (
    <Stats>
      <StatCard
        title="손익 확인 대기"
        value={fmt(r.pending, 0) + '건'}
        detail="후속 시세가 들어오면 손익 확정 · 아직 학습 전"
      />
      <StatCard
        title="학습 가능한 경험"
        value={fmt(r.eligible, 0) + '건'}
        detail={'미학습 ' + fmt(r.untrained, 0) + '건'}
      />
      <StatCard
        title="학습 완료"
        value={fmt(completed, 0) + '건'}
        detail="완료 누계 · DB 잔여와 다름"
      />
      <StatCard
        title="Replay DB"
        value={bytes(r.bytes)}
        badge={
          <Badge tone={num(r.quarantined) + num(r.unsupported) ? 'warn' : 'neutral'}>
            {num(r.quarantined) + num(r.unsupported) ? '보류 있음' : '사용 가능'}
          </Badge>
        }
        detail={
          '현재 ' +
          fmt(r.total, 0) +
          '건 · 보류 ' +
          fmt(r.quarantined, 0) +
          ' · 호환 불가 ' +
          fmt(r.unsupported, 0)
        }
      />
    </Stats>
  );
}
export function ResourceCards({ status: s }: { status: StatusResponse }) {
  const gpu = obj(s.physical_gpu),
    used = num(gpu.memory_used_mb),
    total = num(gpu.memory_total_mb);
  return (
    <Stats>
      <StatCard
        title="GPU · 전체 VRAM"
        value={fmt(used / 1024) + ' / ' + fmt(total / 1024) + ' GiB'}
        badge={<Badge>{str(s.gpu, 'GPU')}</Badge>}
        progress={total ? (used / total) * 100 : 0}
        detail={'GPU 연산 ' + fmt(gpu.utilization_percent, 0) + '% · 다른 앱 포함'}
      />
      <StatCard
        title="시장 Feed"
        value={s.feed_running ? '수신 프로세스 실행' : '정지'}
        badge={
          <Badge tone={s.feed_running ? 'good' : 'neutral'}>{s.feed_running ? 'ON' : 'OFF'}</Badge>
        }
        detail={
          '최근 5분 ' +
          fmt(at(s, 'input_availability.fresh'), 0) +
          ' / ' +
          fmt(s.configured_instruments, 0) +
          '종목'
        }
      />
      <StatCard
        title="Agent · 모델 프로세스"
        value={s.agent_process_running ? '실행 중' : '정지'}
        detail={
          'Champion ' +
          state(s.model_runtime?.champion?.status) +
          ' / Candidate ' +
          state(s.model_runtime?.candidate?.status)
        }
      />
    </Stats>
  );
}
export function SummaryTable({ data, fields }: { data: Data; fields: [string, string][] }) {
  return (
    <Table
      headers={['항목', '현재 값']}
      rows={fields.map(([key, text]) => [
        text,
        <Display key={key} value={at(data, key)} field={key.split('.').at(-1)} />,
      ])}
    />
  );
}
