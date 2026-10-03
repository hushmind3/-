import { useMemo, useState } from 'react';
import { at, fmt, list, obj, rows, state, str, time } from '../data';
import type { StatusResponse } from '../types';
import {
  FillTable,
  LearningBoard,
  ModeControls,
  ModelCards,
  ResourceCards,
  ReplayCards,
  SystemControls,
} from '../components/operations';
import {
  Badge,
  Button,
  Buttons,
  Card,
  DataPanel,
  Panel,
  StatCard,
  Stats,
  Table,
} from '../components/ui';
export function Operations({ status: s }: { status: StatusResponse }) {
  const runtime = s.model_runtime || {},
    errs = Object.entries(runtime).filter(([, r]) => r?.error);
  const replay = obj(runtime.champion?.replay);
  const trial = obj(s.validation_comparison);
  return (
    <>
      <Stats>
        <StatCard
          title="현재 운영"
          value={errs.length ? '모델 오류' : s.feed_running ? '시장 수신 중' : '시스템 정지'}
          badge={
            <Badge tone={errs.length ? 'bad' : s.feed_running ? 'good' : 'neutral'}>
              {errs.length ? '확인 필요' : s.feed_running ? 'Feed ON' : 'OFF'}
            </Badge>
          }
          detail={
            'Champion ' +
            state(runtime.champion?.status || 'stopped') +
            ' · Candidate ' +
            state(runtime.candidate?.status || 'stopped')
          }
        />
        <StatCard
          title="최근 상태 조회"
          value={time(s.status_updated_at)}
          detail="현재 실행과 마지막 저장 기록을 구분해 표시"
        />
      </Stats>
      <LearningBoard status={s} />
      <Stats>
        <StatCard
          title="승급 시험"
          value={state(trial.status)}
          detail={
            fmt(trial.bars_current, 0) +
            ' / ' +
            fmt(trial.bars_required, 0) +
            '구간 · 장기 운영계좌 유지'
          }
        />
      </Stats>
      <ReplayCards replay={replay} />
      <div className="two-column">
        <ModeControls status={s} />
        <SystemControls status={s} />
      </div>
      <ModelCards status={s} />
      <Panel title="최근 주문 · 체결">
        <FillTable fills={at(s, 'paper_account.fills')} />
      </Panel>
      <ResourceCards status={s} />
      <DataPanel title="현재 적용 설정 · 입력 범위" data={s.input_availability} />
      <DataPanel title="운영 계좌 비교 · 비용 기준" data={s.live_account_comparison} />
      <DataPanel title="장기 목표 · 누적 승리 기록" data={s.paper_account} />
      <DataPanel
        title="최근 실행 오류 · 과거 기록"
        data={{
          current: errs.map(([role, r]) => ({ role, error: r?.error })),
          history: {
            candidate: at(s, 'metrics.candidate_update_error'),
            champion: at(s, 'metrics.champion_update_error'),
          },
        }}
      />
    </>
  );
}
export function Markets({ status: s }: { status: StatusResponse }) {
  const [search, setSearch] = useState(''),
    [fresh, setFresh] = useState(false),
    [market, setMarket] = useState('all');
  const instruments = useMemo(
    () =>
      rows(s.instruments).filter(
        (i) =>
          (!fresh || i.fresh) &&
          (market === 'all' ||
            list(rows(s.markets).find((m) => m.key === market)?.symbols).includes(
              i.symbol || '',
            )) &&
          [i.symbol, i.name, i.market].some((v) =>
            str(v, '').toLowerCase().includes(search.toLowerCase()),
          ),
      ),
    [s.instruments, s.markets, search, fresh, market],
  );
  return (
    <>
      <ResourceCards status={s} />
      <Card title="시장별 수신 현황">
        <Table
          headers={['시장', '종목 수', '최근 수신', '상태']}
          rows={rows(s.markets).map((m) => [
            <Button onClick={() => setMarket(str(m.key))}>{str(m.label || m.key)}</Button>,
            fmt(m.count ?? list(m.symbols).length, 0),
            fmt(m.fresh_count, 0) + '종목 수신 · ' + time(m.latest_decision),
            str(m.status || m.session),
          ])}
        />
      </Card>
      <Card title="종목 · 시세 · 최근 판단">
        <Buttons>
          <Button className={market === 'all' ? 'active' : ''} onClick={() => setMarket('all')}>
            전체 시장
          </Button>
          {rows(s.markets).map((m) => (
            <Button
              key={str(m.key)}
              className={market === m.key ? 'active' : ''}
              onClick={() => setMarket(str(m.key))}
            >
              {str(m.label)}
            </Button>
          ))}
        </Buttons>
        <div className="filters">
          <label>
            종목 검색
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="종목명 / 코드 / 시장"
            />
          </label>
          <label className="checkbox">
            <input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} />
            최근 5분 수신만
          </label>
          <span>{instruments.length}종목</span>
        </div>
        <Table
          headers={['종목', '이름', '시장', '최근 시세', '가격', '거래량', '판단', '목표 비중']}
          rows={instruments.map((i) => {
            const q = obj(i.quote),
              d = obj(i.decision);
            return [
              str(i.symbol),
              str(i.name),
              str(i.market),
              time(q.date),
              fmt(q.close, 4),
              fmt(q.volume, 0),
              str(d.action),
              str(d.target_weight),
            ];
          })}
        />
      </Card>
      <DataPanel title="호가 · 체결 입력 · 시세 해상도" data={s.input_availability} />
      <DataPanel title="현재 모델 판단 · 행동 확률" data={s.decisions} />
      <DataPanel title="추가 시장 입력 · universe 확장" data={s.universe_expansion} />
    </>
  );
}
