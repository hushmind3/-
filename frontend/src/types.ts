export type Json = string | number | boolean | null | Json[] | Data;
export interface Data {
  [key: string]: Json | undefined;
}
export type Role = 'champion' | 'candidate';
export interface ApiResult extends Data {
  ok?: boolean;
  error?: string;
  message?: string;
}
export interface PaperBook extends Data {
  initial_cash?: number;
  cash?: number;
  equity?: number;
  net_pnl?: number;
  net_return_rate?: number;
  fees?: number;
  slippage?: number;
  spread?: number;
  sell_tax?: number;
  costs?: number;
  realized_pnl?: number;
  unrealized_pnl?: number;
  trade_count?: number;
  positions?: Data | Data[];
  marks?: Data;
}
export interface LearningState extends Data {
  loss?: number;
  reward_points?: number;
  updated_at?: string;
  samples?: number;
  seconds?: number;
}
export interface ReplayState extends Data {
  total?: number;
  eligible?: number;
  untrained?: number;
  pending?: number;
  quarantined?: number;
  unsupported?: number;
  bytes?: number;
  daily?: Data[];
}
export interface Decision extends Data {
  action?: string;
  target_weight?: number;
  current_weight?: number;
  cash_weight?: number;
  as_of?: string;
  value?: number;
  seconds?: number;
}
export interface ExpertRecord extends Data {
  id?: string;
  name?: string;
  role?: string;
  parameters?: number;
  dtype?: string;
  checkpoint_bytes?: number;
  weight_bytes?: number;
  ram_bytes?: number;
  vram_bytes?: number;
  location?: string;
  loaded?: boolean;
  active?: boolean;
  frozen?: boolean;
  router_selected?: boolean;
  last_inference_seconds?: number;
  last_used_at?: string;
  last_timings?: Data;
  input_shapes?: Data;
  output_shape?: Json;
  universe?: string[] | null;
}
export interface RegistryTotals extends Data {
  expert_count?: number;
  parameters?: number;
  active_parameters?: number;
  checkpoint_bytes?: number;
  weight_bytes?: number;
  ram_bytes?: number;
  vram_bytes?: number;
}
export interface Instrument extends Data {
  symbol?: string;
  name?: string;
  market?: string;
  fresh?: boolean;
  quote?: Data;
  decision?: Decision | null;
}
export interface Market extends Data {
  key?: string;
  label?: string;
  symbols?: string[];
  count?: number;
  fresh_count?: number;
  session?: string;
}
export interface ModelRuntime extends Data {
  status?: string;
  loaded?: boolean;
  requested?: boolean;
  learning_active?: boolean;
  optimizer_updates?: number;
  books?: { [currency: string]: PaperBook };
  replay?: ReplayState;
  learning?: LearningState;
  ram_weight_bytes?: number;
  gpu_weight_bytes?: number;
  compute_device?: string;
  last_decision?: string;
  decision_seconds?: number;
  source_kind?: string;
}
export interface StatusResponse extends Data {
  running?: boolean;
  feed_running?: boolean;
  observe_enabled?: boolean;
  paper_enabled?: boolean;
  learning_enabled?: boolean;
  real_orders_enabled?: boolean;
  model_runtime?: { champion?: ModelRuntime; candidate?: ModelRuntime };
  metrics?: Data;
  learning?: Data;
  instruments?: Instrument[];
  markets?: Market[];
  physical_gpu?: {
    utilization_percent?: number;
    memory_used_mb?: number;
    memory_total_mb?: number;
  };
  logs?: string | Json[];
}
export interface WorkerStatus extends ModelRuntime {
  alive?: boolean;
  stop_requested?: boolean;
  error?: string;
  decision?: Decision;
  fills?: Data[];
  compute?: Data;
  checkpoint_bytes?: number;
  parameters?: number;
}
export interface Recipe extends Data {
  candidate_id?: string;
  parent_id?: string;
  enabled_experts?: string[];
  refresh_seconds?: Data;
  evaluation_state?: string;
  mutation_description?: string;
  scores?: Data;
}
export interface AssemblyResponse extends ApiResult {
  enabled?: boolean;
  settings?: Data;
  champion?: Recipe;
  candidate?: Recipe;
  queue?: Recipe[];
  history?: Data[];
  experts?: ExpertRecord[];
  new_experts?: string[];
  worker?: WorkerStatus;
}
export interface RegistryResponse extends Data {
  experts?: ExpertRecord[];
  totals?: RegistryTotals;
  pipeline?: Data;
  unavailable?: Data[];
}
export interface ProviderStatus extends Data {
  saved?: boolean;
  environment?: string;
  last_test?: Data;
}
export interface Responses {
  '/api/status': StatusResponse;
  '/api/trading-moe/status': WorkerStatus;
  '/api/assembly/status': AssemblyResponse;
  '/api/experts': RegistryResponse;
  '/api/provider': ProviderStatus;
  '/api/runtime': Data;
  '/api/health': Data;
}
export type ReadEndpoint = keyof Responses;
