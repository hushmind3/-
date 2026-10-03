import { fmt, num, obj, state, str, time } from '../data';
import type { StatusResponse } from '../types';
import { Books } from '../components/operations';
import { Badge, Card, DataPanel, StatCard, Stats } from '../components/ui';
export function Trial({ status: s }: { status: StatusResponse }) {
  const t = obj(s.validation_comparison),
    count = num(t.bars_current),
    required = num(t.bars_required);
  return (
    <>
      <Stats>
        <StatCard
          title="고정 모델 승급 시험"
          value={state(t.status)}
          badge={
            <Badge tone={t.active ? 'good' : 'neutral'}>{t.active ? '시험 중' : '시험 중지'}</Badge>
          }
          detail={str(t.reason)}
        />
        <StatCard
          title="시험 진행"
          value={fmt(count, 0) + ' / ' + fmt(required, 0) + '구간'}
          progress={required ? (count / required) * 100 : 0}
          detail={time(t.last_timestamp)}
        />
        <StatCard
          title="동일 조건 비교"
          value={t.comparison_valid ? '비교 가능' : '비교 대기'}
          detail={
            'Champion ' +
            str(t.champion_snapshot_version) +
            ' / Candidate ' +
            str(t.snapshot_version)
          }
        />
      </Stats>
      <p className="notice">
        이 화면은 기존 고정 모델 승급 시험입니다. 장기 운영계좌는 유지됩니다. TradingMoE 후보 실험은{' '}
        <a href="#assembly">조립 · 자동실험</a>에서 확인하세요.
      </p>
      <Card title="Champion · 승급 시험계좌">
        <Books books={t.champion} />
      </Card>
      <Card title="Candidate · 승급 시험계좌">
        <Books books={t.candidate} />
      </Card>
      <DataPanel title="현재 판단 · 입력 · 거래비용 · 비교 규칙" data={t} />
      <DataPanel title="하루 시험 · 승격/탈락 기록" data={s.daily_cycle} />
      <DataPanel title="학습 예선 · 승급 기준" data={s.learning} />
    </>
  );
}
