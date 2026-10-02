"use strict";

// This screen reads the actual TradingMoE registry. No model roster or weight
// values live in the frontend. Stable cards update individual fields only.
const expertCards = new Map();
let expertRequest = null;
let expertFusionPending = false;
let expertPipeline = null;
function expertSeconds(value) {
  return value == null ? "미측정" : decimal(value, 3) + "초";
}
function expertBytes(value) {
  if (value == null) return "측정 불가";
  const bytes = Number(value);
  return bytes >= 1024 ** 3
    ? (bytes / 1024 ** 3).toFixed(2) + " GiB"
    : (bytes / 1024 ** 2).toFixed(2) + " MiB";
}
function expertSet(node, value) {
  const next = String(value ?? "—");
  if (node.textContent !== next) node.textContent = next;
}
function expertTone(node, label, tone = "") {
  expertSet(node, label);
  const next = "pill " + tone;
  if (node.className !== next) node.className = next;
}
function createExpertCard(entry) {
  const card = document.createElement("article");
  card.className = "expert-card";
  card.dataset.expertId = entry.id;
  card.innerHTML = `<header><h2 data-field="name"></h2><span data-field="active" class="pill"></span></header>
    <p data-field="role"></p><strong class="expert-parameters" data-field="parameters"></strong>
    <div class="expert-badges"><span data-field="dtype" class="pill"></span><span data-field="frozen" class="pill"></span>
      <span data-field="loaded" class="pill"></span><span data-field="router" class="pill"></span></div>
    <dl>${[["location", "현재 위치"], ["checkpoint", "원본 checkpoint"], ["weights", "실질 가중치"],
      ["ram", "현재 worker RAM"], ["vram", "현재 tensor VRAM"], ["coldLoad", "첫 적재 · CPU"],
      ["gpuTransfer", "가중치 GPU 전송"], ["forward", "Forward · 원본 계산"], ["roundTrip", "전체 왕복"],
      ["lastUsed", "최근 사용 시각"], ["input", "입력 shape"], ["output", "출력 shape"]]
      .map(([key, label]) => `<div><dt>${label}</dt><dd data-field="${key}"></dd></div>`).join("")}</dl>
    <p data-field="error" role="alert" hidden></p>
    <details class="compact-detail"><summary>원본 출력 보기 · 변환 전</summary>
      <p data-field="rawOrigin"></p><button type="button" data-field="rawLoad">원본 출력 불러오기</button>
      <a data-field="rawDownload" download>전체 원본 JSON 저장</a>
      <pre data-field="raw" class="expert-raw" tabindex="0">아직 불러오지 않았습니다.</pre></details>`;
  const fields = Object.fromEntries([...card.querySelectorAll("[data-field]")].map(n => [n.dataset.field, n]));
  const state = {card, fields, entry, rawKey:null, pending:false};
  fields.rawDownload.href = "/api/experts/output?id=" + encodeURIComponent(entry.id);
  fields.rawDownload.download = entry.id + ".raw.json";
  fields.rawLoad.addEventListener("click", () => loadExpertRaw(state));
  expertCards.set(entry.id, state);
  return state;
}
async function loadExpertRaw(state) {
  if (state.pending) return;
  state.pending = true;
  state.fields.rawLoad.disabled = true;
  expertSet(state.fields.rawLoad, "불러오는 중");
  try {
    const response = await api("/api/experts/output?id=" + encodeURIComponent(state.entry.id));
    expertSet(state.fields.raw, JSON.stringify(response.packet.native_output, null, 2));
    expertSet(state.fields.rawOrigin,
      (response.origin === "runtime_inference" ? "TradingMoE 실제 추론" : "독립 추론 검증") +
      " · " + (response.packet.input_authenticity?.startsWith("synthetic")
        ? "합성 입력 · 추론 경로 검증용" : "입력 기준 " + (response.packet.as_of || "시장 시각 없음")) + " · " + response.packet.units +
      " · shape " + response.packet.output_shape.join(" × "));
    state.rawKey = state.entry.last_used_at || "independent";
    expertSet(state.fields.rawLoad, "원본 출력 다시 불러오기");
  } catch (error) {
    expertSet(state.fields.raw, "불러오기 실패: " + error.message);
    expertSet(state.fields.rawLoad, "다시 시도");
  } finally {
    state.pending = false;
    state.fields.rawLoad.disabled = false;
  }
}
function renderExperts(data) {
  expertPipeline = data.pipeline || {};
  const p = expertPipeline;
  const paper = p.paper_trading;
  $("moePaperSection").hidden = !paper;
  if (paper) {
    const book = paper.books?.USD || {};
    expertTone($("moePaperBadge"), (paper.status === "paper_complete" ? "구간 실행 완료" : "가상매매 실행 중") + " · 실제 주문 OFF", "good");
    expertSet($("moePaperNav"), decimal(book.equity, 2) + " USD");
    expertSet($("moePaperPnl"), "누적 손익 " + decimal(book.net_pnl, 2) + " USD · 수수료 " + decimal(book.fees, 2));
    expertSet($("moePaperCash"), decimal(book.cash, 2) + " USD");
    expertSet($("moePaperPositions"), Object.entries(book.positions || {}).map(([s,v]) => s + " " + (s === "ETHUSDT" ? decimal(v.quantity * .001, 3) + " ETH" : v.quantity + "개")).join(" · ") || "보유 없음");
    expertSet($("moePaperFills"), whole(book.trade_count || 0) + "건");
    expertSet($("moePaperReward"), "계좌 보상 점수 " + decimal(paper.reward_points?.USD, 4));
    expertSet($("moePaperUpdates"), whole(paper.optimizer_updates || 0) + "회");
    expertSet($("moePaperTime"), "최근 전체 사이클 " + expertSeconds(paper.cycle_seconds));
    expertSet($("moePaperDecision"), Object.entries(paper.actions || {}).map(([s,a]) => s + " " + a + (paper.tradable_symbols && !paper.tradable_symbols.includes(s) ? " · 현재 체결 시세 없음 / 참고 의견" : " · 목표 " + decimal(100 * (paper.target_weights?.[s] || 0), 1) + "%")).join(" / "));
    expertSet($("moePaperTimestamp"), "과거 시장 입력 " + paper.timestamp + " · 다음 실제 관측 봉에서 체결 · 1계약=0.001 ETH · USDT=USD 가상 평가");
    expertSet($("moePaperOrders"), (paper.fills || []).map(f => f.date + " " + f.symbol + " " + f.action + " " + (f.symbol === "ETHUSDT" ? decimal(f.quantity * .001, 3) + " ETH @ " + decimal(f.price * 1000, 4) : f.quantity + "개 @ " + decimal(f.price, 4)) + " · 수수료 " + decimal(f.fee, 4)).join("\n") || "아직 체결 없음");
  }
  const stages = {input_adapter:"입력 변환 중", router:"전문가 선택 중", experts:"전문가 추론 중",
    output_adapter:"원본 출력 변환 중", fusion:"공통 fusion 계산 중", complete:"통합 추론 완료", error:"통합 추론 오류"};
  expertTone($("expertPipelineBadge"), stages[p.stage] || "실행 기록 없음", p.stage === "complete" ? "good" : p.stage === "error" ? "bad" : "");
  expertSet($("expertPipelineProgress"), p.selected_experts?.length
    ? whole(p.completed_experts || 0) + " / " + p.selected_experts.length + "개 완료" : "미실행");
  expertSet($("expertPipelineSummary"), p.stage === "complete"
    ? "전체 " + expertSeconds(p.timings?.total_seconds) + " · 출력 변환 " + expertSeconds(p.timings?.adapter_seconds) +
      " · CPU fusion " + expertSeconds(p.timings?.fusion_seconds) + " · " + timeOf(p.completed_at) + (paper ? " · 가상 체결 / 손익 학습 연결" : " · 단독 추론 기록") +
      (p.as_of ? " · 입력 기준 " + p.as_of : "") +
      (p.input_authenticity?.includes("synthetic") ? " · 일부 합성 입력으로 경로 검증" : "")
    : p.error || "선택한 전문가를 순차 실행하고 원본 출력과 통합 출력을 각각 보관합니다.");
  $("expertFusionLoad").disabled = expertFusionPending || p.stage !== "complete";
  const totals = data.totals || {};
  const summaries = [
    ["expert_count", "등록된 전문가", data.registered ? whole(totals.expert_count) + "개" : "등록 안 됨"],
    ["parameters", "전체 파라미터", data.registered ? (totals.parameters / 1e9).toFixed(6) + " B" : "—"],
    ["checkpoint_bytes", "전체 원본 checkpoint", data.registered ? expertBytes(totals.checkpoint_bytes) : "—"],
    ["ram_bytes", "현재 expert worker RAM", data.registered ? expertBytes(totals.ram_bytes) : "—"],
    ["vram_bytes", "현재 expert tensor VRAM", data.registered ? expertBytes(totals.vram_bytes) : "—"],
    ["active_parameters", "현재 실행 중 파라미터", data.registered ? (totals.active_parameters / 1e9).toFixed(6) + " B" : "—"],
  ];
  if (!$("expertTotals").children.length) {
    $("expertTotals").innerHTML = summaries.map(([key, label]) =>
      `<article class="live-metric"><div class="metric-heading">${label}</div><strong id="expertTotal-${key}" class="metric-number">—</strong></article>`).join("");
  }
  for (const [key, , value] of summaries) expertSet($("expertTotal-" + key), value);
  expertTone($("expertRegistryBadge"), data.registered ? (paper ? "expert 고정 · controller 학습" : "원본 고정 · 학습 안 함") : "TradingMoE 등록 안 됨", data.registered ? "good" : "warn");
  expertSet($("expertRegistrySummary"), data.registered
    ? "현재 사용량은 이 TradingMoE worker만 집계합니다. Windows·다른 앱의 GPU 사용량과 검증 때의 peak는 포함하지 않습니다. 추론이 끝나면 worker 종료 → disk 보관으로 돌아갑니다."
    : "독립 추론 검증 후 등록된 전문가가 이 화면에 자동 표시됩니다.");
  const ids = new Set((data.experts || []).map(e => e.id));
  for (const [id, state] of expertCards) if (!ids.has(id)) {
    state.card.remove();
    expertCards.delete(id);
  }
  for (const entry of data.experts || []) {
    let state = expertCards.get(entry.id);
    if (!state) {
      state = createExpertCard(entry);
      $("expertCards").append(state.card);
    }
    state.entry = entry;
    const f = state.fields;
    expertSet(f.name, entry.name);
    expertSet(f.role, entry.role);
    expertSet(f.parameters, whole(entry.parameters) + "개 · " + (entry.parameters / 1e9).toFixed(6) + " B");
    expertTone(f.dtype, entry.dtype);
    expertTone(f.active, entry.active ? "실행 중 · active" : "실행 안 함 · inactive", entry.active ? "good" : "");
    expertTone(f.frozen, entry.frozen ? "가중치 고정 · frozen" : "학습 가능 · trainable", entry.frozen ? "good" : "warn");
    expertTone(f.loaded, entry.loaded ? "적재됨 · loaded" : "미적재 · unloaded", entry.loaded ? "good" : "");
    expertTone(f.router, entry.router_selected ? "최근 router 선택" : "router 선택 안 됨", entry.router_selected ? "good" : "");
    expertSet(f.location, entry.location);
    expertSet(f.checkpoint, entry.checkpoint_present ? expertBytes(entry.checkpoint_bytes) : "원본 파일 누락");
    expertSet(f.weights, expertBytes(entry.weight_bytes) + " · parameter × dtype");
    expertSet(f.ram, expertBytes(entry.ram_bytes));
    expertSet(f.vram, expertBytes(entry.vram_bytes));
    const timing = entry.last_timings || {};
    for (const [field, key] of [["coldLoad", "cold_load_seconds"], ["gpuTransfer", "gpu_transfer_seconds"],
      ["forward", "forward_seconds"], ["roundTrip", "round_trip_seconds"]]) expertSet(f[field], expertSeconds(timing[key]));
    expertSet(f.lastUsed, entry.last_used_at ? timeOf(entry.last_used_at) : "TradingMoE에서 아직 미사용");
    expertSet(f.input, Object.entries(entry.input_shapes).map(([key, shape]) => key + " " + shape.join(" × ")).join(" / "));
    expertSet(f.output, entry.output_shape.join(" × "));
    f.error.hidden = !entry.error;
    expertSet(f.error, entry.error);
    if (state.rawKey && state.rawKey !== (entry.last_used_at || "independent")) {
      expertSet(f.rawOrigin, "새 추론 결과가 있습니다. 다시 불러오면 최신 원본 출력을 확인합니다.");
    }
  }
  html("expertUnavailable", () => (data.unavailable || []).map(e =>
    `<p><strong>${esc(e.name)}</strong> · 공식 공개 경로에서 학습된 checkpoint 미확보 · 등록 제외</p>`).join(""));
  $("expertRegistryError").hidden = !data.recent_error;
  expertSet($("expertRegistryError"), data.recent_error);
}
async function loadExpertFusion() {
  if (expertFusionPending) return;
  expertFusionPending = true;
  const button = $("expertFusionLoad");
  button.disabled = true;
  expertSet(button, "불러오는 중");
  try {
    const result = await api("/api/experts/fusion");
    expertSet($("expertFusionOutput"), JSON.stringify({run_id:result.run_id, as_of:result.as_of,
      selected_experts:result.selected_experts, adapter_status:result.adapter_status,
      fusion_status:result.fusion_output.status, fusion_shapes:result.fusion_output.shapes,
      trading_output:result.trading_output, pipeline_timings:result.pipeline_timings}, null, 2));
    expertSet(button, "통합 결과 다시 불러오기");
  } catch (error) {
    expertSet($("expertFusionOutput"), "불러오기 실패: " + error.message);
    expertSet(button, "다시 시도");
  } finally {
    expertFusionPending = false;
    button.disabled = expertPipeline?.stage !== "complete";
  }
}
$("expertFusionLoad").addEventListener("click", loadExpertFusion);
function refreshExperts() {
  if (expertRequest) return expertRequest;
  expertRequest = (async () => {
    try {
      renderExperts(await api("/api/experts"));
      expertTone($("systemBadge"), "서버 연결됨", "good");
      return true;
    } catch (error) {
      expertTone($("expertRegistryBadge"), "registry 읽기 실패", "bad");
      $("expertRegistryError").hidden = false;
      expertSet($("expertRegistryError"), error.message + " · 표시된 수치는 마지막 수신값입니다.");
      return false;
    } finally { expertRequest = null; }
  })();
  return expertRequest;
}
