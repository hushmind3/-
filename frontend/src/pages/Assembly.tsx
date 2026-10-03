import { api } from '../api';
import { usePolling } from '../hooks';
import { at, bytes, fmt, list, num, obj, pct, state, str, time } from '../data';
import {
  ActionButton,
  Badge,
  Buttons,
  Card,
  DataPanel,
  ErrorState,
  Loading,
  Panel,
  StatCard,
  Stats,
  Table,
  Toggle,
} from '../components/ui';
export function Assembly() {
  const q = usePolling('/api/assembly/status', 3000),
    a = q.data;
  if (!a) return q.error ? <ErrorState message={q.error} retry={q.refresh} /> : <Loading />;
  const c = a.candidate || {},
    ch = a.champion || {},
    settings = obj(a.settings),
    worker = obj(a.worker),
    newIds = list(a.new_experts).map((v) => (typeof v === 'string' ? v : str(obj(v).id))),
    selected = new Set(list(c.enabled_experts).map((v) => str(v))),
    champ = new Set(list(ch.enabled_experts).map((v) => str(v)));
  return (
    <>
      {q.error && <ErrorState message={q.error} retry={q.refresh} />}
      <Stats>
        <StatCard
          title="자동 조립"
          value={a.enabled ? '실행 중' : '정지'}
          badge={<Badge tone={a.enabled ? 'good' : 'neutral'}>{a.enabled ? 'ON' : 'OFF'}</Badge>}
          detail={str(a.message)}
        />
        <StatCard
          title="현재 Champion"
          value={str(ch.candidate_id)}
          detail={'세대 ' + fmt(a.generation, 0)}
        />
        <StatCard
          title="현재 Candidate"
          value={str(c.candidate_id)}
          badge={<Badge>{state(c.evaluation_state)}</Badge>}
        />
        <StatCard
          title="대기 / 누적 시험"
          value={(a.queue || []).length + ' / ' + fmt(a.experiments, 0)}
          detail={
            '승격 ' +
            fmt(a.promotions, 0) +
            ' · 탈락 ' +
            fmt(a.rejections, 0) +
            ' · 새 expert ' +
            newIds.length
          }
        />
      </Stats>
      <Card title="조립 · 시험 제어">
        <Buttons>
          <ActionButton label="자동조립 시작" path={api.assembly('start')} disabled={!!a.enabled} />
          <ActionButton label="자동조립 정지" path={api.assembly('stop')} disabled={!a.enabled} />
          <ActionButton label="새 Candidate 즉시 생성" path={api.assembly('generate')} />
          <ActionButton
            label="현재 Candidate 시험 시작"
            path={api.assembly('trial/start')}
            disabled={!a.candidate || !!worker.alive}
          />
          <ActionButton
            label="현재 Candidate 시험 정지"
            path={api.assembly('trial/stop')}
            disabled={!worker.alive}
          />
          <ActionButton
            label="현재 Candidate 탈락 · 다음 후보"
            path={api.assembly('next')}
            tone="warn"
          />
        </Buttons>
        <Buttons>
          <Toggle
            label="자동 Candidate 교체"
            enabled={!!settings.auto_replace}
            path={api.assembly('settings')}
            field="auto_replace"
          />
          <Toggle
            label="자동 승격"
            enabled={!!settings.auto_promote}
            path={api.assembly('settings')}
            field="auto_promote"
          />
          <Toggle
            label="새 Expert 자동감지"
            enabled={!!settings.detect_experts}
            path={api.assembly('settings')}
            field="detect_experts"
          />
        </Buttons>
        <p className="muted">
          큰 PT는 공용으로 유지합니다. 후보마다 작은 recipe와 학습 state만 저장합니다.
        </p>
      </Card>
      <Card title="Expert 조립표">
        <Table
          headers={[
            'Expert',
            'Champion',
            'Candidate',
            '역할',
            '학습 universe',
            '최근 선택',
            '갱신 주기',
          ]}
          rows={(a.experts || []).map((e) => {
            const id = str(e.id),
              universe = list(e.universe);
            return [
              <>
                {str(e.name)} {newIds.includes(id) && <Badge tone="good">NEW</Badge>}
              </>,
              <Badge tone={champ.has(id) ? 'good' : 'neutral'}>
                {champ.has(id) ? '사용' : 'OFF'}
              </Badge>,
              <Badge tone={selected.has(id) ? 'good' : 'neutral'}>
                {selected.has(id) ? '사용' : 'OFF'}
              </Badge>,
              e.role === 'market' ? '시장 인식' : '매매 정책',
              universe.length ? (
                <Panel
                  title={
                    universe.length +
                    '종목 · ' +
                    universe
                      .slice(0, 3)
                      .map((v) => str(v))
                      .join(', ')
                  }
                >
                  {universe.map((v) => str(v)).join(', ')}
                </Panel>
              ) : (
                '범용 시장 입력'
              ),
              worker.assembly_candidate_id === c.candidate_id &&
              list(worker.selected_experts).includes(id)
                ? '선택됨'
                : '—',
              'Champion ' +
                fmt(at(ch, 'refresh_seconds.' + id), 0) +
                ' / 후보 ' +
                fmt(at(c, 'refresh_seconds.' + id), 0) +
                '초',
            ];
          })}
        />
      </Card>
      <Card title="현재 Candidate · 변경 내용" badge={<Badge>{state(c.evaluation_state)}</Badge>}>
        <p>{str(c.mutation_description)}</p>
        <p className="muted">
          {str(
            c.reason ||
              (worker.assembly_candidate_id === c.candidate_id ? worker.message : undefined),
            '시험 대기',
          )}
        </p>
        <p className="muted">
          부모 {str(c.parent_id)} · 생성 {time(c.created_at)} · 전체 PT 복제{' '}
          {fmt(a.checkpoint_copies, 0)}개 · 학습 state {bytes(a.candidate_state_bytes)}
        </p>
        <Table
          headers={[
            '단계',
            '후보 수익률',
            'Champion',
            '차이',
            '체결',
            '비용',
            '최대 손실폭',
            '실행시간',
          ]}
          rows={Object.entries(obj(c.scores)).map(([phase, v]) => {
            const score = obj(v),
              candidate = obj(score.candidate),
              champion = obj(score.champion);
            return [
              phase === 'replay' ? 'Replay 예선' : 'Paper 비교',
              pct(candidate.net_return),
              pct(champion.net_return),
              pct(score.delta),
              fmt(candidate.trades, 0),
              fmt(num(candidate.fees) + num(candidate.slippage), 4),
              pct(candidate.max_drawdown),
              fmt(candidate.seconds, 3) + '초',
            ];
          })}
        />
        <DataPanel title="Recipe · Router · Policy · 비교 성적 전체" data={c} />
      </Card>
      <Card title="다음 Candidate 대기열">
        <Table
          headers={['후보', '부모', '변경 내용', '평가 상태', '생성']}
          rows={(a.queue || []).map((r) => [
            str(r.candidate_id),
            str(r.parent_id),
            str(r.mutation_description),
            state(r.evaluation_state),
            time(r.created_at),
          ])}
        />
      </Card>
      <Card title="과거 실험 · 승격/탈락 이유">
        <Table
          headers={['시각', '후보', '결과', '변경 내용', '이유', 'Paper 차이']}
          rows={(a.history || [])
            .slice(-50)
            .reverse()
            .map((h) => [
              time(h.time || h.timestamp || h.updated_at || h.created_at),
              str(h.candidate_id || at(h, 'recipe.candidate_id')),
              state(h.event || h.evaluation_state || h.state),
              str(h.mutation),
              str(h.reason || h.message),
              pct(at(h, 'scores.paper.delta')),
            ])}
        />
      </Card>
      <DataPanel
        title="시험 Worker · 적용 대기 · history 전체"
        data={{
          worker: a.worker,
          history: a.history,
          champion: ch,
          registry_versions: a.registry_versions,
        }}
      />
    </>
  );
}
