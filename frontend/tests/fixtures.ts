import type {
  Data,
  StatusResponse,
  WorkerStatus,
  AssemblyResponse,
  RegistryResponse,
} from '../src/types';
// Small response fixtures matching the existing Python API, not model/training fixtures.
export const book = {
  initial_cash: 10000,
  cash: 9000,
  equity: 10010,
  net_pnl: 10,
  fees: 1,
  trade_count: 2,
  positions: { ETHUSDT: { quantity: 630, average_cost: 1.58 } },
  marks: { ETHUSDT: 1.6 },
};
export const status: StatusResponse = {
  running: false,
  feed_running: false,
  agent_process_running: false,
  observe_enabled: true,
  paper_enabled: true,
  learning_enabled: true,
  real_orders_enabled: false,
  status_updated_at: '2026-10-03T08:00:00Z',
  configured_instruments: 2,
  model_runtime: {
    champion: {
      status: 'stopped',
      loaded: false,
      optimizer_updates: 7,
      learning: { loss: 0.1, updated_at: '2026-10-03T08:00:00Z' },
      replay: { total: 2, eligible: 1, pending: 1, bytes: 1024 },
      books: { USD: book },
    },
    candidate: { status: 'stopped', loaded: false, optimizer_updates: 3, books: { USD: book } },
  },
  instruments: [
    {
      symbol: 'MSFT',
      name: 'Microsoft',
      market: 'us',
      fresh: true,
      quote: { close: '400', volume: '10', date: '2026-10-03T08:00:00Z' },
    },
    {
      symbol: '005930.KS',
      name: '삼성전자',
      market: 'kr',
      fresh: false,
      quote: { close: '80000', volume: '20', date: '2026-10-02T08:00:00Z' },
    },
  ],
  markets: [
    { key: 'us', label: '미국 주식', count: 1, symbols: ['MSFT'], fresh_count: 1 },
    { key: 'kr', label: '한국 주식', count: 1, symbols: ['005930.KS'], fresh_count: 0 },
  ],
  physical_gpu: { utilization_percent: 7, memory_used_mb: 1000, memory_total_mb: 8192 },
  gpu: 'RTX 3070',
  input_availability: { fresh: 1, configured: 2 },
  learning: {
    multiscale_input_status: {
      '1m': {
        mean_history_coverage: 0.9,
        observed_symbols: 2,
        available_symbols: 2,
        complete_history_symbols: 0,
      },
    },
    replay_current: 2,
  },
  metrics: {
    champion_last_completed_round: { timeframe_samples: { '1m': 128 } },
    candidate_last_completed_round: { samples: 64, timeframe_samples: { '1m': 64 } },
  },
  validation_comparison: {
    status: 'paused',
    bars_current: 1,
    bars_required: 390,
    champion: { USD: book },
    candidate: { USD: book },
  },
  logs: 'server ready\nmodels stopped',
  paper_account: {
    fills: [
      { symbol: 'MSFT', action: 'BUY', quantity: 2, price: 400, date: '2026-10-03T08:00:00Z' },
    ],
  },
};
export const moe: WorkerStatus = {
  status: 'stopped',
  alive: false,
  parameters: 2228312033,
  checkpoint_bytes: 8000000000,
  expert_count: 20,
  optimizer_updates: 7,
  load_count: 1,
  checkpoint: 'champion.pt',
  books: { USD: book },
  decision: {
    action: 'HOLD',
    target_weight: 0.5,
    current_weight: 0.1,
    cash_weight: 0.5,
    seconds: 0.2,
    as_of: '2023-02-01T00:00:00Z',
  },
  learning: { loss: 0.1, reward_points: -0.01 },
  compute: { inference_device: 'cuda:0', learning_device: 'cuda:0', allocated_bytes: 1024 },
  replay: { total: 1, eligible: 1, pending: 1, bytes: 1024 },
  fills: [
    {
      symbol: 'ETHUSDT',
      action: 'BUY',
      quantity: 759,
      price: 1.583198304,
      fee: 1.2,
      date: '2023-02-01T00:00:00Z',
    },
  ],
};
export const assembly: AssemblyResponse = {
  ok: true,
  enabled: false,
  settings: { auto_replace: true, auto_promote: false, detect_experts: true },
  experiments: 5,
  promotions: 0,
  rejections: 4,
  checkpoint_copies: 0,
  candidate_state_bytes: 1024,
  champion: { candidate_id: 'champion-base', enabled_experts: ['test_expert'] },
  candidate: {
    candidate_id: 'asm-test',
    parent_id: 'champion-base',
    evaluation_state: 'ready',
    enabled_experts: ['test_expert'],
    refresh_seconds: { test_expert: 60 },
  },
  queue: [],
  new_experts: ['test_expert'],
  experts: [
    { id: 'test_expert', name: 'Registry Dynamic Expert', role: 'policy', universe: ['MSFT'] },
  ],
  history: [
    { candidate_id: 'old', event: 'rejected', reason: 'score lower', time: '2026-10-03T08:00:00Z' },
  ],
  worker: { status: 'stopped', alive: false },
};
export const registry: RegistryResponse = {
  experts: [
    {
      id: 'test_expert',
      name: 'Registry Dynamic Expert',
      parameters: 1000,
      dtype: 'FP32',
      frozen: true,
      loaded: false,
      active: false,
      location: 'disk',
      checkpoint_bytes: 4096,
      weight_bytes: 4000,
      ram_bytes: 0,
      vram_bytes: 0,
    },
  ],
  totals: {
    expert_count: 1,
    parameters: 1000,
    checkpoint_bytes: 4096,
    weight_bytes: 4000,
    ram_bytes: 0,
    vram_bytes: 0,
  },
  unavailable: [],
};
export const responses: Record<string, Data> = {
  '/api/status': status,
  '/api/trading-moe/status': moe,
  '/api/assembly/status': assembly,
  '/api/experts': registry,
  '/api/provider': { saved: true, environment: 'real', last_test: { ok: true } },
  '/api/runtime': { gpu_scheduler: { active: null, waiting: [] } },
  '/api/health': { ok: true },
  '/api/experts/fusion': { ok: true, trading_output: { action: 'HOLD' } },
  '/api/experts/output?id=test_expert': { ok: true, raw: [0.1, 0.2] },
  '/api/provider/public-ip': { ok: true, ip: '127.0.0.1' },
};
