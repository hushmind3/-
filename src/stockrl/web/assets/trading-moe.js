"use strict";

// Dedicated screen: small worker status only, no model loading or expert API.
let moeStatusRequest = null;
let moeCommandPending = false;
let moeFillKey = null;
let moeCommandError = null;
function renderTradingMoE(data) {
  const labels = {stopped:"정지", loading:"PT 로딩 중", running:"실행 중", saving:"저장 중", error:"중지됨 · 오류"};
  badge("moeRunStatus", labels[data.status] || "정지", data.status === "running" ? "good" : data.error ? "bad" : "");
  badge("systemBadge", "서버 연결됨", "good");
  $("moeStart").disabled = moeCommandPending || data.alive;
  $("moeStop").disabled = moeCommandPending || !data.alive || data.stop_requested;
  text("moeRunMeta", "TradingMoE.pt · " + decimal((data.parameters || 2227295521) / 1e9, 3) + "B · " + decimal((data.checkpoint_bytes || 0) / 1e9, 2) + "GB · PID " + (data.pid || "—") + " · PT 적재 " + (data.load_count || 0) + "회");
  const compute=data.compute || {};
  text("moeCompute", compute.gpu_name ? compute.gpu_name + " · 전문가 추론 " + compute.inference_device + " · 판단/학습 " + compute.learning_device + (data.alive ? " · GPU 상주 tensor " + decimal((compute.allocated_bytes || 0)/1024**2,1) + " MiB · worker RAM " + decimal((data.worker_ram_bytes || 0)/1024**3,2) + " GiB · expert 계산 중에는 VRAM이 증가합니다 · GPU 동시 expert 1개" : " · worker 종료 · GPU 메모리 해제됨") : "CUDA 장치 확인 중");
  text("moeRunMessage", data.error || moeCommandError || (data.stop_requested ? "정지 요청됨 · 현재 사이클 후 상태를 저장하고 종료합니다." : data.message || "공식 ETHUSDT 과거 시세에서 이어 실행합니다."));
  const book = data.books?.USD || {};
  const nav = book.equity;
  text("moeNav", accountMoney(nav, "USD"));
  text("moeReturn", "수익률 " + accountPercent(nav == null ? null : nav / book.initial_cash - 1));
  text("moeCash", accountMoney(book.cash, "USD"));
  text("moePnl", accountMoney(book.net_pnl, "USD"));
  text("moeFees", "총 수수료 " + accountMoney(book.fees, "USD"));
  const position = book.positions?.ETHUSDT;
  const mark = book.marks?.ETHUSDT;
  const exposure = position && nav ? position.quantity * mark / nav : 0;
  text("moeExposure", "ETH 현재 비중 " + accountPercent(exposure));
  text("moePosition", position ? decimal(position.quantity * .001, 3) + " ETH" : "보유 없음");
  text("moePositionPrices", position ? "평단 " + decimal(position.average_cost * 1000, 2) + " / 현재가 " + decimal(mark * 1000, 2) + " USD" : "보유 수량 0 · 현금 유지");
  text("moePositionPnl", position ? "평가손익 " + accountMoney(position.quantity * (mark - position.average_cost), "USD") + " · 비중 " + accountPercent(exposure) : "평가손익 0 USD");
  const decision = data.decision || {};
  text("moeAction", decision.action ? "ETHUSDT  " + decision.action : "첫 판단 대기");
  badge("moeDecisionBadge", decision.action || "판단 대기", decision.action === "BUY" ? "good" : "");
  text("moeWeights", "현재 " + accountPercent(decision.current_weight) + " · 배분 의견 " + accountPercent(decision.target_weight) + " · 현금 의견 " + accountPercent(decision.cash_weight) + (decision.action === "HOLD" ? " · 이번 행동: 보유 유지" : ""));
  $("moeTargetProgress").value = 100 * (decision.target_weight || 0);
  text("moeDecisionDetail", "과거 시세 " + (decision.as_of || "—") + " · value " + (decision.value == null ? "—" : decimal(decision.value, 5)) + " · 판단 " + (decision.seconds == null ? "—" : decimal(decision.seconds, 3) + "초"));
  for (const node of document.querySelectorAll("#moeVerticalFlow [data-stage]")) node.classList.toggle("active", data.status === "running" && !!data.stages?.[node.dataset.stage]);
  const learning = data.learning || {};
  text("moeReward", learning.reward_points == null ? "—" : decimal(learning.reward_points, 5));
  text("moeRewardTotal", "누적 계좌 보상 " + decimal(data.reward_points?.USD || 0, 5));
  text("moeLoss", learning.loss == null ? "—" : decimal(learning.loss, 6));
  text("moeLearnTime", "최근 학습 " + (learning.updated_at ? timeOf(learning.updated_at) : "—"));
  text("moeUpdates", whole(data.optimizer_updates || 0) + "회");
  text("moeReplay", "학습 대기 " + whole(Math.max(0,(data.replay?.untrained || 0)-(data.applied_replay_rows || 0))) + " · 학습 완료 / 저장 대기 " + whole(Math.min(data.replay?.untrained || 0,data.applied_replay_rows || 0)) + " · 결과 대기 " + whole(data.replay?.pending || 0));
  badge("moeFillCount", whole(book.trade_count || 0) + "건", "");
  const fills = data.fills || [];
  const key = JSON.stringify(fills);
  if (key !== moeFillKey) {
    $("moeFillRows").innerHTML = fills.length ? fills.slice(-20).reverse().map(f => "<tr>" + [f.date, f.symbol, f.action, decimal(f.quantity * .001, 3) + " ETH", decimal(f.price * 1000, 2), decimal(f.fee, 4), decimal(f.realized_pnl, 4)].map(value => "<td>" + esc(value) + "</td>").join("") + "</tr>").join("") : '<tr><td colspan="7">아직 체결 없음</td></tr>';
    moeFillKey = key;
  }
}
async function refreshTradingMoE() {
  if (moeStatusRequest) return moeStatusRequest;
  moeStatusRequest = (async () => {
    try { renderTradingMoE(await api("/api/trading-moe/status")); return true; }
    catch (error) { badge("moeRunStatus", "상태 조회 실패", "bad"); text("moeRunMessage", error.message); return false; }
    finally { moeStatusRequest = null; }
  })();
  return moeStatusRequest;
}
async function commandTradingMoE(action) {
  if (moeCommandPending) return;
  moeCommandPending = true;
  moeCommandError = null;
  $("moeStart").disabled = $("moeStop").disabled = true;
  text("moeRunMessage", action === "start" ? "시작 요청 중" : "정지 요청 중");
  try {
    const response = await api("/api/trading-moe/" + action, {});
    if (!response.ok) throw new Error(response.error || "실행 요청 실패");
    renderTradingMoE(response.state);
  } catch (error) { moeCommandError="요청 실패: " + error.message; text("moeRunMessage", moeCommandError); }
  finally { moeCommandPending = false; await refreshTradingMoE(); }
}
$("moeStart").addEventListener("click", () => commandTradingMoE("start"));
$("moeStop").addEventListener("click", () => commandTradingMoE("stop"));
setInterval(() => { if (!document.hidden && screenVisible("trading-moe")) refreshTradingMoE(); }, 1000);
