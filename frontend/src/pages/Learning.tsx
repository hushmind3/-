import { at, fmt, obj, pct, str } from '../data';
import type { StatusResponse } from '../types';
import { LearningBoard, ReplayCards } from '../components/operations';
import { Badge, Card, DataPanel, Display, Stats, StatCard, Table } from '../components/ui';
export function Learning({ status: s }: { status: StatusResponse }) {
  const l = obj(s.learning),
    m = obj(s.metrics),
    r = obj(s.model_runtime?.champion?.replay);
  return (
    <>
      <LearningBoard status={s} />
      <ReplayCards
        replay={
          Object.keys(r).length
            ? r
            : {
                total: l.replay_current,
                eligible: l.eligible_backlog,
                untrained: l.replay_untrained_count,
                pending: l.replay_pending_count,
                quarantined: l.replay_quarantined_count,
                unsupported: l.replay_unsupported_count,
                bytes: l.replay_bytes,
                daily: l.daily_learning,
              }
        }
      />
      <Stats>
        <StatCard
          title="최근 학습 샘플"
          value={
            fmt(
              at(m, 'candidate_last_completed_round.samples') ?? l.candidate_training_samples,
              0,
            ) + '건'
          }
          detail="최근 완료한 학습 회차"
        />
        <StatCard
          title="최근 학습 소요"
          value={fmt(l.candidate_update_seconds ?? m.candidate_update_seconds, 3) + '초'}
          detail={str(l.candidate_skip_reason, '대기 사유 없음')}
        />
        <StatCard
          title="Replay 잔여"
          value={fmt(l.replay_current, 0) + '건'}
          badge={<Badge>{s.learning_enabled ? '학습 허용' : '학습 OFF'}</Badge>}
          detail={
            '보류 ' +
            fmt(l.replay_quarantined_count, 0) +
            '건 · 결과 대기 ' +
            fmt(l.replay_pending_count, 0) +
            '건'
          }
        />
      </Stats>
      <Card title="시간봉별 실제 입력 확보">
        <Table
          headers={[
            '시간봉',
            '입력 확보율',
            '입력 보유 / 관측',
            '기록 충족 종목',
            'Champion 사용',
            'Candidate 사용',
          ]}
          rows={Object.entries(obj(l.multiscale_input_status)).map(([key, v]) => {
            const d = obj(v);
            return [
              key,
              pct(d.mean_history_coverage),
              fmt(d.available_symbols, 0) + ' / ' + fmt(d.observed_symbols, 0),
              fmt(d.complete_history_symbols, 0),
              <Display value={at(m, 'champion_last_completed_round.timeframe_samples.' + key)} />,
              <Display value={at(m, 'candidate_last_completed_round.timeframe_samples.' + key)} />,
            ];
          })}
        />
        <p className="muted">
          기록 충족은 필요한 전체 입력 길이를 채운 종목입니다. 일부 기록이 있는 종목도 학습에 사용할
          수 있습니다.
        </p>
      </Card>
      <DataPanel title="날짜별 경험 생성 · 학습 완료 내역" data={l.daily_learning || r.daily} />
      <DataPanel
        title="보류 경험 · 사유 · DB 경로"
        data={{
          blocked: l.blocked_replay,
          reasons: r.blocked_reasons,
          db: l.replay_db_path,
          replay: r,
        }}
      />
      <DataPanel title="보상 기준 · 손익 평가 기간" data={l.reward_credit} />
      <DataPanel title="10배 목표 · 계좌 진행" data={s.paper_account} />
      <DataPanel title="모델별 학습 · optimizer · 샘플 상세" data={l} />
      <DataPanel title="모델 구조 · 실제 계산 시간 · 체크포인트" data={m} />
    </>
  );
}
