import { useState } from 'react';
import { api } from '../api';
import { usePolling } from '../hooks';
import { at, bytes, fmt, num, obj, str, time } from '../data';
import type { Data } from '../types';
import {
  ActionButton,
  Badge,
  Card,
  DataPanel,
  DataTree,
  ErrorState,
  Loading,
  Panel,
  StatCard,
  Stats,
  Table,
} from '../components/ui';
function ExpertCard({ expert: e }: { expert: Data }) {
  const [raw, setRaw] = useState<Data>();
  return (
    <Card
      title={str(e.name)}
      badge={
        <Badge tone={e.error ? 'bad' : e.active ? 'good' : 'neutral'}>
          {e.active ? '실행 중' : e.loaded ? '적재됨' : '미적재'}
        </Badge>
      }
    >
      <strong className="stat compact">
        {fmt(num(e.parameters) / 1e9, 4)}B <small>{str(e.dtype)}</small>
      </strong>
      <p>{str(e.role)}</p>
      <dl className="key-values">
        <div>
          <dt>원본 Checkpoint / 가중치 메모리</dt>
          <dd>
            {bytes(e.checkpoint_bytes)} / {bytes(e.weight_bytes)}
          </dd>
        </div>
        <div>
          <dt>현재 위치 / RAM / VRAM</dt>
          <dd>
            {str(e.location)} / {bytes(e.ram_bytes)} / {bytes(e.vram_bytes)}
          </dd>
        </div>
        <div>
          <dt>원본 가중치 / Router</dt>
          <dd>
            {e.frozen ? '고정' : '학습 가능'} · {e.router_selected ? '최근 선택됨' : '미선택'}
          </dd>
        </div>
        <div>
          <dt>최근 추론 / 사용 시각</dt>
          <dd>
            {fmt(e.last_inference_seconds, 3)}초 · {time(e.last_used_at)}
          </dd>
        </div>
      </dl>
      {e.error && <p className="bad">{str(e.error)}</p>}
      <Panel title="적재 · GPU 전송 · 계산 · 왕복 시간">
        <Table
          headers={['첫 적재', 'GPU 전송', '계산', '전체 왕복']}
          rows={[
            [
              ...[
                'cold_load_seconds',
                'gpu_transfer_seconds',
                'forward_seconds',
                'round_trip_seconds',
              ].map((k) => fmt(at(e, 'last_timings.' + k), 3) + '초'),
            ],
          ]}
        />
      </Panel>
      <Panel title="원본 입력 · 출력 shape · raw 출력">
        <a
          href={'/api/experts/output?id=' + encodeURIComponent(str(e.id))}
          download={str(e.id) + '.raw.json'}
        >
          전체 원본 JSON 저장
        </a>
        <DataTree
          data={{
            input_shapes: e.last_input_shapes || e.input_shapes,
            output_shape: e.last_output_shape || e.output_shape,
            origin: e.raw_output_origin,
            sha256: e.raw_output_sha256,
          }}
        />
        <ActionButton
          label={str(e.name) + ' 원본 출력 조회'}
          task={async () => {
            const r = await api.raw(str(e.id));
            setRaw(r);
            return r;
          }}
        />
        {raw && <DataTree data={raw} />}
      </Panel>
      <DataPanel title="학습 universe · 출처 · 라이선스 · 원본 메타데이터" data={e} />
    </Card>
  );
}
export function Experts() {
  const q = usePolling('/api/experts'),
    r = q.data;
  const [fusion, setFusion] = useState<Data>();
  if (!r) return q.error ? <ErrorState message={q.error} retry={q.refresh} /> : <Loading />;
  const t = obj(r.totals);
  return (
    <>
      {q.error && <ErrorState message={q.error} retry={q.refresh} />}
      <Stats>
        <StatCard
          title="등록된 Expert"
          value={fmt(t.expert_count, 0) + '개'}
          detail="실제 Registry 목록"
        />
        <StatCard
          title="전체 파라미터"
          value={fmt(num(t.parameters) / 1e9, 4) + 'B'}
          detail={'활성 ' + fmt(num(t.active_parameters) / 1e9, 4) + 'B'}
        />
        <StatCard
          title="Checkpoint 합계"
          value={bytes(t.checkpoint_bytes)}
          detail={'가중치 ' + bytes(t.weight_bytes)}
        />
        <StatCard
          title="Expert RAM / VRAM"
          value={bytes(t.ram_bytes) + ' / ' + bytes(t.vram_bytes)}
          detail="현재 적재된 전문가 기준"
        />
      </Stats>
      <Card title="통합 추론 · 원본 출력 유지">
        <a href="/api/experts/fusion" download="TradingMoE.json">
          전체 통합 JSON 저장
        </a>
        <ActionButton
          label="최근 통합 출력 조회"
          task={async () => {
            const result = await api.fusion();
            setFusion(result);
            return result;
          }}
        />
        {fusion && <DataTree data={fusion} />}
        <DataPanel title="최근 통합 경로 · 가상매매 · 계좌 변경" data={r.pipeline} />
      </Card>
      <div className="two-column">
        {(r.experts || []).map((e) => (
          <ExpertCard key={str(e.id)} expert={e} />
        ))}
      </div>
      <DataPanel title="사용 불가 모델 · 원인" data={r.unavailable} />
      <DataPanel
        title="Router · Fusion · 학습 연결 상태"
        data={{
          router: r.router_status,
          fusion: r.fusion_head_status,
          live: r.live_integration,
          error: r.recent_error,
        }}
      />
    </>
  );
}
