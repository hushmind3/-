import type { Data, Json } from './types';
export function obj(value: Json | undefined): Data {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : {};
}
export function list(value: Json | undefined): Json[] {
  return Array.isArray(value) ? value : [];
}
export function rows(value: Json | undefined): Data[] {
  return list(value).map(obj);
}
export function at(value: Data | undefined, path: string): Json | undefined {
  return path.split('.').reduce<Json | undefined>((v, k) => obj(v)[k], value);
}
export function num(value: Json | undefined, fallback = 0): number {
  const n = Number(value);
  return value == null || !Number.isFinite(n) ? fallback : n;
}
export function str(value: Json | undefined, fallback = '—'): string {
  return value == null
    ? fallback
    : typeof value === 'object'
      ? JSON.stringify(value)
      : String(value);
}
export function fmt(value: Json | undefined, digits = 2): string {
  return value == null
    ? '—'
    : num(value).toLocaleString('ko-KR', { maximumFractionDigits: digits });
}
export function pct(value: Json | undefined): string {
  return value == null ? '—' : fmt(num(value) * 100) + '%';
}
export function bytes(value: Json | undefined): string {
  if (value == null) return '—';
  const n = num(value);
  return n >= 1024 ** 3
    ? fmt(n / 1024 ** 3) + ' GiB'
    : n >= 1024 ** 2
      ? fmt(n / 1024 ** 2) + ' MiB'
      : fmt(n, 0) + ' B';
}
export function time(value: Json | undefined): string {
  if (!value) return '기록 없음';
  const d = new Date(str(value));
  return Number.isNaN(d.getTime()) ? str(value) : d.toLocaleString('ko-KR', { hour12: false });
}
export const stateLabels: Record<string, string> = {
  generated: '생성',
  installed: '장착',
  registry_changed: 'Expert 감지',
  replay_testing: 'Replay 시험 중',
  paper_testing: 'Paper 시험 중',
  validating: '시험 중',
  collecting: '입력 수집 중',
  awaiting_promotion: '승격 대기',
  stopped: '정지',
  loading: '로딩 중',
  running: '실행 중',
  saving: '저장 중',
  waiting: '대기',
  ready: '시험 대기',
  queued: '대기열',
  replay: 'replay 예선',
  paper: 'paper 비교',
  qualified: '통과 · 승격 대기',
  promoted: '승격',
  rejected: '탈락',
  blocked: '보류',
  paused: '중지',
  complete: '완료',
  observing: '판단 중',
  catching_up: '처리 중',
  error: '오류',
};
export function state(value: Json | undefined): string {
  return stateLabels[str(value)] || str(value);
}
export const pages = [
  ['control', '운영 · 계좌'],
  ['markets', '시장 · 판단'],
  ['learning', '경험 학습'],
  ['promotionTrial', '승급전'],
  ['connection', '연결 설정'],
  ['system', '상세 · 기록'],
  ['experts', 'TradingMoE · 전문가'],
  ['trading-moe', 'TradingMoE · 자동매매'],
  ['assembly', '조립 · 자동실험'],
] as const;
export type Page = (typeof pages)[number][0];
const aliases: Record<string, Page> = {
  overview: 'control',
  market: 'markets',
  trial: 'promotionTrial',
  details: 'system',
  portfolio: 'control',
  promotion: 'promotionTrial',
};
export function pageFromHash(hash: string): Page {
  const key = hash.replace(/^#/, '');
  return pages.some(([p]) => p === key) ? (key as Page) : aliases[key] || 'control';
}
export const fieldLabels: Record<string, string> = {
  equity: '순자산',
  cash: '현금',
  net_pnl: '비용 차감 손익',
  initial_cash: '시작 자금',
  fees: '수수료',
  slippage: '슬리피지',
  spread: '스프레드 비용',
  sell_tax: '매도 세금',
  realized_pnl: '실현손익',
  unrealized_pnl: '평가손익',
  trade_count: '체결 수',
  holdings_value: '보유 평가액',
  positions: '포지션',
  books: '통화별 계좌',
  marks: '현재 평가 가격',
  quantity: '수량',
  average_cost: '평균 매입가',
  symbol: '종목',
  currency: '통화',
  action: '행동',
  target_weight: '목표 비중',
  cash_weight: '현금 목표 비중',
  confidence: '확신도',
  probabilities: '행동 확률',
  value: 'Controller 가치',
  total: '전체',
  eligible: '학습 가능',
  untrained: '학습 대기',
  pending: '손익 확인 대기',
  quarantined: '학습 불가 보류',
  unsupported: '호환되지 않는 경험',
  bytes: 'DB 용량',
  remaining: '잔여',
  completed: '학습 완료',
  enqueued: '생성',
  exposures: '학습 사용',
  optimizer_updates: '가중치 업데이트',
  loss: '최근 loss',
  reward_points: '보상 점수',
  updated_at: '최근 갱신',
  last_timestamp: '최근 데이터 시점',
  date: '시각',
  last_update_utc: '최근 학습 시각',
  candidate_stage: 'Candidate 학습 단계',
  candidate_skip_reason: '학습 대기 이유',
  candidate_update_seconds: '최근 학습 소요 초',
  candidate_training_samples: '최근 학습 샘플',
  candidate_optimizer_steps: '최근 optimizer 횟수',
  replay_current: 'Replay 잔여',
  replay_untrained_count: '미학습 경험',
  replay_quarantined_count: '보류 경험',
  replay_unsupported_count: '호환 불가 경험',
  replay_pending_count: '결과 대기',
  multiscale_input_status: '시간봉별 입력 확보',
  observed_symbols: '관측 종목',
  available_symbols: '입력 보유 종목',
  complete_history_symbols: '기록 충족 종목',
  mean_history_coverage: '평균 입력 확보율',
  status: '상태',
  reason: '이유',
  parameters: '파라미터',
  dtype: 'dtype',
  checkpoint_bytes: 'Checkpoint 용량',
  ram_bytes: 'RAM',
  vram_bytes: 'VRAM',
  location: '적재 위치',
  loaded: '적재 여부',
  active: '실행 여부',
  frozen: '원본 가중치 고정',
  router_selected: 'Router 선택',
  last_used_at: '최근 사용 시각',
  input_shapes: '원본 입력 shape',
  output_shape: '원본 출력 shape',
  cold_load_seconds: '첫 적재 시간',
  gpu_transfer_seconds: 'GPU 전송 시간',
  forward_seconds: '계산 시간',
  round_trip_seconds: '왕복 시간',
  bars_current: '시험 진행',
  bars_required: '필요 구간',
  comparison_valid: '동일 조건 비교 가능',
  same_market_input: '같은 시장 입력',
  same_cost_rules: '같은 비용 조건',
  snapshot_version: 'Candidate 고정 버전',
  champion_snapshot_version: 'Champion 고정 버전',
  max_drawdown: '최대 손실폭',
  seconds: '소요 시간',
  trades: '체결',
  net_return: '비용 차감 수익률',
  initial_NAV: '최초 NAV',
  final_NAV: '최종 NAV',
  pnl: '손익',
  delta: 'Champion 대비 차이',
  created_at: '생성 시각',
  parent_id: '부모',
  candidate_id: '후보 ID',
  mutation_description: '변경 내용',
  evaluation_state: '시험 단계',
  base_checkpoint_hash: '기준 가중치 hash',
  enabled_experts: '사용 expert',
  symbol_applicability: '학습 universe',
  expert_roles: '역할',
  native_inputs: 'Native 입력 규격',
  market_routing: '시장 Router',
  policy_routing: '정책 Router',
  refresh_seconds: '출력 갱신 주기',
  trainable_state: '작은 학습 state',
  reward_credit: '손익 평가 기간/보상 기준',
  daily_learning: '날짜별 학습',
  blocked_reasons: '보류 이유',
  source_kind: '현재 입력 종류',
  source: '데이터 출처',
  learning_active: '실제 학습 중',
  decision_seconds: '판단 소요 초',
  last_decision: '최근 판단 데이터 시점',
  compute_device: '계산 장치',
  ram_weight_bytes: '모델 RAM',
  gpu_weight_bytes: '모델 VRAM',
  checkpoint: '모델 파일',
  error: '최근 오류',
  learning: '학습',
  replay: '경험 DB',
};
export const label = (key: string) => fieldLabels[key] || key;
