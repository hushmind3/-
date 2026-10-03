"use strict";

// Operating overview and market screen. Account/learning/trial renderers live with their screen.
const optimizerObservations = new Map();

function renderOptimizerBoard(d, prefix = "overview") {
  const roles = moeModelRoles(d);
  const updates = roles.map((role) => d.model_runtime[role]);
  if (!updates.length || d.status_unavailable) {
    badge(prefix + "OptimizerBadge", d.status_unavailable ? "연결 끊김" : "MoE 정지", "");
    text(prefix + "OptimizerCount", "—");
    text(prefix + "OptimizerChange", "등록된 MoE 학습 기록 없음");
    text(prefix + "OptimizerLoss", "—"); text(prefix + "OptimizerReward", "—");
    text(prefix + "OptimizerTime", "모델 시작 후 실제 업데이트 표시");
    optimizerObservations.clear();
    return;
  }
  let changed = false;
  const deltas = roles.map((role) => {
    const count = d.model_runtime[role].optimizer_updates;
    if (count == null) return "업데이트 수 미확인";
    let previous = optimizerObservations.get(role);
    if (!previous || count < previous.count) previous = {baseline:count,count};
    changed ||= count > previous.count;
    optimizerObservations.set(role,{baseline:previous.baseline,count});
    return (role === "champion" ? "Champion" : "Candidate") + " +" + whole(count - previous.baseline);
  });
  badge(prefix + "OptimizerBadge", !roles.some((role) => modelIsRunning(d,role)) ? "모델 정지 · 마지막 기록" : d.learning_enabled === false ? "학습 중지"
    : roles.some((role) => modelIsLearning(d,role)) ? "최근 사이클 학습" : "학습 대기", changed ? "good" : "");
  text(prefix + "OptimizerCount", updates.every((r) => r.optimizer_updates != null)
    ? whole(updates.reduce((n,r) => n + r.optimizer_updates,0)) + "회 누적" : "미확인");
  text(prefix + "OptimizerChange", "이 화면 확인 이후 · " + deltas.join(" / "));
  const latest = updates.filter((r) => r.learning?.updated_at)
    .sort((a,b) => String(b.learning.updated_at).localeCompare(String(a.learning.updated_at)))[0];
  text(prefix + "OptimizerLoss", decimal(latest?.learning?.loss,6));
  text(prefix + "OptimizerReward", decimal(latest?.learning?.reward_points,4));
  text(prefix + "OptimizerTime", latest ? "마지막 학습 완료 " + timeOf(latest.learning.updated_at)
    + " · " + ageOf(latest.learning.updated_at) : "아직 완료된 학습 기록 없음");
  if (changed) {
    const node = $(prefix + "OptimizerCount");
    if (node) { node.classList.remove("value-updated"); requestAnimationFrame(() => node.classList.add("value-updated")); }
  }
}

function marketFor(symbol, markets) {
  return (markets || []).find((g) => (g.symbols || []).includes(symbol));
}

function marketName(group) {
  return (
    {
      "global:Japan": "일본 지수",
      "global:HongKong": "홍콩 지수",
      "global:Germany": "독일 지수",
      "global:UK": "영국 지수",
    }[group.key] ||
    group.label ||
    group.key
  );
}

function action(value) {
  const label =
    { BUY: "매수", HOLD: "관망", SELL: "매도" }[value] || value || "—";
  return '<span class="action ' + esc(value) + '">' + esc(label) + "</span>";
}

function renderStatus(d) {
  const f = d.feed_metrics || {},
    m = d.metrics || {},
    fresh = (f.fresh_symbols_5m || []).length;
  badge(
    "feedBadge",
    d.feed_running ? (fresh ? "수신 중" : "수집기 실행") : "수집기 정지",
    d.feed_running ? (fresh ? "good" : "warn") : "bad",
  );
  text(
    "freshCount",
    whole(fresh) + " / " + whole(d.configured_instruments) + "개",
  );
  text(
    "feedDetail",
    d.feed_running
      ? "최근 5분 새 시세 · 저장 " +
          whole(d.feed_rows) +
          "행 · " +
          ageOf(f.updated_at_utc)
      : "시세 수집기가 정지돼 있습니다.",
  );
  const roles = runningModelRoles(d);
  const transition = Object.values(d.model_runtime || {}).find((r) => ["loading", "saving"].includes(r.status));
  badge("agentBadge", transition ? modelStateLabel(transition)
    : roles.length ? (d.observe_enabled === false ? "실행 중 · 판단 중지" : "실행 중") : "모델 정지",
    transition ? "warn" : roles.length ? "good" : "");
  text("decisionCount", roles.length + " / 2개 실행");
  text("agentDetail", roles.length ? roles.map((role) => {
    const r = d.model_runtime?.[role] || {};
    return (role === "champion" ? "Champion" : "Candidate") + " · "
      + (modelUsesHistoricalData(d, role) ? "ETHUSDT 과거 가상매매 · 데이터 " + marketDataTime(r.last_decision)
        : "실시간 시세 판단 · " + timeOf(r.last_decision || m.last_market_timestamp));
  }).join(" / ") : "모델 미실행 · 시세 수집과 별도 제어");

  const p = d.provider || {},
    last = f.broker_last_message_utc,
    connected = !!f.broker_connected,
    auth = !!p.last_test?.ok;
  let btitle, bdetail, btone;
  if (!p.saved) {
    btitle = "키 미등록";
    bdetail = "키움 키를 연결해야 한국 체결 시세를 받습니다.";
    btone = "warn";
  } else if (!auth) {
    btitle = "인증 미확인";
    bdetail = p.last_test?.message || "저장된 키를 재검사해 주세요.";
    btone = "warn";
  } else if (!connected) {
    btitle = "소켓 미연결";
    bdetail = f.broker_error || "키 인증은 완료, 실시간 연결 대기 중";
    btone = "warn";
  } else if (!recent(last)) {
    btitle = "체결 대기";
    bdetail = "인증·소켓 연결 완료 · 최근 체결 메시지 없음";
    btone = "warn";
  } else {
    btitle = "체결 수신 중";
    bdetail =
      "최근 체결 " + ageOf(last) + " · 구독 " + whole(f.broker_symbols) + "개";
    btone = "good";
  }
  badge(
    "brokerBadge",
    connected ? (recent(last) ? "체결 수신" : "연결 · 체결 대기") : "소켓 미연결",
    btone,
  );
  text("brokerState", btitle);
  text("brokerDetail", bdetail);
}

function renderMarkets(d) {
  const all = d.markets || [],
    f = d.feed_metrics || {};
  text(
    "marketSummary",
    "등록 " +
      whole(d.configured_instruments) +
      "종목 · 최근 5분 수신 " +
      whole((f.fresh_symbols_5m || []).length) +
      "종목",
  );
  html(
    "marketGrid",
    () =>
      all
        .map((g) => {
          const n = num(g.fresh_count),
            registered = num(g.count),
            off = String(g.session || "").includes("장외"),
            tone = n ? "good" : off ? "" : "warn";
          return (
            '<button type="button" class="market clickable" data-market="' +
            esc(g.key) +
            '"><div class="head"><span class="name">' +
            esc(marketName(g)) +
            '</span><span class="pill ' +
            tone +
            '">' +
            (n
              ? "수신 중"
              : off
                ? "장외 | 다음 세션 시세 대기"
                : "새 시세 없음") +
            '</span></div><div class="count">' +
            whole(n) +
            " <small>/ 등록 " +
            whole(registered) +
            '</small></div><div class="session">' +
            esc(g.session || "거래 시간 확인 중") +
            "</div></button>"
          );
        })
        .join("") || '<div class="card pad">등록된 시장이 없습니다.</div>',
  );
  const has = (key) => all.some((g) => g.key === key && g.count > 0);
  html("marketGaps", () =>
    [
      ["korea_index", "한국 지수"],
      ["korea_bonds", "한국 국채·채권"],
      ["korea_futures", "한국 지수선물"],
    ]
      .map(
        ([key, label]) =>
          '<span class="market-gap">' +
          label +
          " · " +
          (has(key) ? "등록됨" : "현재 데이터 미등록") +
          "</span>",
      )
      .join(""),
  );
}

function renderDecisions(d) {
  text(
    "championVersion",
    "현재 champion · " + (d.champion_version || "확인 중"),
  );

  const renderKey = JSON.stringify([
    selectedMarket,
    freshOnly,
    $("symbolSearch").value,
    d.metrics?.last_market_timestamp,
    Math.floor(Date.now() / 30000),
    d.instruments?.map((r) => [
      r.symbol,
      r.fresh,
      r.quote?.close,
      r.quote?.date,
      r.decision?.action,
      r.decision?.p_sell,
      r.decision?.p_hold,
      r.decision?.p_buy,
      r.decision?.value,
      r.decision?.date,
    ]),
  ]);
  if (renderKey === lastDecisionRenderKey) return;
  lastDecisionRenderKey = renderKey;
  const all = d.markets || [],
    groups = [["all", "전체"], ...all.map((g) => [g.key, marketName(g)])];
  html("marketFilters", () =>
    groups
      .map(
        ([key, label]) =>
          '<button type="button" data-filter="' +
          esc(key) +
          '" class="' +
          (selectedMarket === key ? "active" : "") +
          '">' +
          esc(label) +
          "</button>",
      )
      .join(""),
  );
  toggleClass("freshOnlyBtn", "active", freshOnly);
  attribute("freshOnlyBtn", "aria-pressed", String(freshOnly));
  const allRows = d.instruments || [],
    query = $("symbolSearch").value.trim().toLocaleUpperCase("ko-KR");
  const scoped = allRows.filter(
      (r) => selectedMarket === "all" || r.group === selectedMarket,
    ),
    rows = scoped.filter((r) => {
      const name = r.name || SYMBOL_NAMES[r.symbol] || "";
      return (
        (!freshOnly || r.fresh) &&
        (!query ||
          String(r.symbol).toLocaleUpperCase("ko-KR").includes(query) ||
          name.toLocaleUpperCase("ko-KR").includes(query))
      );
    });
  const counts = { BUY: 0, HOLD: 0, SELL: 0 };
  scoped.forEach((r) => {
    const a = r.decision?.action;
    if (a in counts) counts[a]++;
  });
  text("buyCount", whole(counts.BUY));
  text("holdCount", whole(counts.HOLD));
  text("sellCount", whole(counts.SELL));
  text(
    "tableCount",
    "현재 " +
      whole(rows.length) +
      "개 표시 / 선택 시장 등록 " +
      whole(scoped.length) +
      "개 · 판단 기록 " +
      whole(counts.BUY + counts.HOLD + counts.SELL) +
      "개",
  );
  html(
    "decisionRows",
    () =>
      rows
        .map((r) => {
          const q = r.quote || {},
            x = r.decision || {},
            name = r.name || SYMBOL_NAMES[r.symbol] || r.symbol,
            prob = Math.max(num(x.p_sell), num(x.p_hold), num(x.p_buy)),
            group = all.find((g) => g.key === r.group),
            price =
              q.close == null
                ? "—"
                : Number(q.close).toLocaleString("ko-KR", {
                    maximumFractionDigits: 3,
                  }),
            fresh = !!r.fresh,
            latest = !!x.action;
          return (
            '<tr class="' +
            (fresh ? "" : "stale") +
            '"><td>' +
            esc(group ? marketName(group) : r.market || "기타") +
            '</td><td><span class="asset-name">' +
            esc(name) +
            '</span><small class="asset-code">' +
            esc(r.symbol) +
            " · " +
            esc(r.asset_class || "-") +
            (/^(0010S0\.KQ|0161M0\.KQ)$/.test(r.symbol)
              ? " | 원본 심볼 확인 필요"
              : "") +
            '</small></td><td><span class="price">' +
            esc(price) +
            '</span><small class="cell-sub">' +
            esc(
              r.provider === "kraken"
                ? "Kraken"
                : (r.market === "KRX" || r.market === "KOSDAQ") &&
                    r.asset_class === "equity" &&
                    d.provider?.provider === "kiwoom"
                  ? "키움 KRX+NXT"
                  : "Yahoo / 공개 시세",
            ) +
            "</small></td><td>" +
            esc(q.date ? timeOf(q.date) : "시세 없음") +
            '<small class="cell-sub">' +
            esc(q.date ? ageOf(q.date) : "수신 전") +
            " · " +
            (fresh ? "최근 수신" : "현재 새 시세 없음") +
            "</small></td><td>" +
            (latest
              ? action(x.action)
              : '<span class="pill">판단 없음</span>') +
            "</td><td>" +
            esc(
              latest
                ? Math.round(prob * 100) + "% · 가치 " + decimal(x.value, 3)
                : "—",
            ) +
            "</td><td>" +
            esc(x.date ? timeOf(x.date) : "기록 없음") +
            '<small class="cell-sub">' +
            esc(x.date ? ageOf(x.date) : "-") +
            "</small></td></tr>"
          );
        })
        .join("") ||
      '<tr><td class="empty" colspan="7">조건에 맞는 등록 종목이 없습니다. 시장 필터와 최근 수신만 설정을 확인하세요.</td></tr>',
  );
}

function renderVenues(d) {
  const f = d.feed_metrics || {},
    connected = !!d.feed_running && !!f.broker_connected;
  for (const [key, label] of [
    ["krx", "KRX"],
    ["nxt", "NXT"],
  ]) {
    const subscribed = num(
        f["broker_" + key + "_subscribed"] ?? f.broker_symbols,
      ),
      ticks = f["broker_" + key + "_ticks"],
      active = f["broker_" + key + "_active_symbols"],
      last = f["broker_" + key + "_last_message_utc"],
      measured = ticks != null,
      isRecent = recent(last);
    badge(
      key + "Badge",
      !connected
        ? "연결 안 됨"
        : !measured
          ? "측정 준비 중"
          : isRecent
            ? "체결 수신 중"
            : "체결 대기",
      !connected ? "bad" : isRecent ? "good" : "warn",
    );
    text(key + "Count", measured ? whole(ticks) + "건 체결" : "측정값 없음");
    text(
      key + "Detail",
      label +
        " " +
        whole(subscribed) +
        "종목 구독 · " +
        (active == null
          ? "활성 종목 측정 전"
          : whole(active) + "종목 체결 확인") +
        " · " +
        (last ? "마지막 " + ageOf(last) : "수신 기록 없음"),
    );
  }
}

function renderTelemetry(d) {
  text(
    "gpuDevice",
    d.gpu?.name || d.metrics?.cuda_device || d.metrics?.device || "장치 미확인",
  );
  const roles = runningModelRoles(d);
  const running = roles.length > 0;
  const devices = roles.map((role) => String(d.model_runtime?.[role]?.compute_device || d.model_runtime?.[role]?.device || d.metrics?.device || ""));
  const cuda = devices.some((device) => device.startsWith("cuda"));
  const mps = devices.some((device) => device.startsWith("mps"));
  badge(
    "gpuBadge",
    cuda
      ? "CUDA 적용"
      : running
        ? mps
          ? "MPS 적용"
          : "CPU 사용"
        : "모델 정지",
    cuda || (running && mps)
      ? "good"
      : running
        ? "warn"
        : "bad",
  );
  const g = d.physical_gpu || {};
  if (g.utilization_percent != null && g.memory_total_mb) {
    const ratio = Math.min(
      100,
      Math.round((g.memory_used_mb / g.memory_total_mb) * 100),
    );
    text(
      "gpuMemory",
      (g.memory_used_mb / 1024).toFixed(1) +
        " / " +
        (g.memory_total_mb / 1024).toFixed(1) +
        " GB",
    );
    text(
      "gpuPercent",
      "GPU 연산 " + whole(g.utilization_percent) + "% / VRAM " + ratio + "%",
    );
    styleWidth("gpuFill", ratio + "%");
    attribute("gpuMeter", "aria-valuenow", String(ratio));
  } else {
    const m = d.metrics || {};
    const used = num(m.cuda_memory_allocated_bytes),
      total = num(m.cuda_total_memory_bytes),
      ratio = total ? Math.min(100, (used / total) * 100) : 0;
    badge(
      "gpuBadge",
      cuda
        ? "CUDA 적용"
        : mps
          ? "MPS 적용"
          : running
            ? "CPU 사용"
            : "모델 정지",
      cuda || mps ? "good" : running ? "warn" : "bad",
    );
    text(
      "gpuMemory",
      total
        ? (used / 1073741824).toFixed(1) +
            " / " +
            (total / 1073741824).toFixed(1) +
            " GB"
        : mps && m.rss_bytes
          ? "MPS 통합 메모리 · 프로세스 " +
            (num(m.rss_bytes) / 1073741824).toFixed(1) +
            " GB"
          : "측정값 없음",
    );
    text(
      "gpuDevice",
      mps
        ? "Apple Silicon MPS · 통합 메모리"
        : m.cuda_device || m.device || "장치 미확인",
    );
    text(
      "gpuPercent",
      mps ? "CPU·GPU 메모리 공유" : total ? Math.round(ratio) + "% 할당" : "—",
    );
    styleWidth("gpuFill", ratio + "%");
    attribute("gpuMeter", "aria-valuenow", String(Math.round(ratio)));
  }
}

function renderOperationsOverview(d) {
  const m = d.metrics || {},
    health = d.agent_health || {},
    candidate = health.candidate || {},
    schedule = m.gpu_scheduler || {};
  const lag = (v) => (v == null ? "미측정" : whole(v) + "초");
  text(
    "opChampionLag",
    !modelIsRunning(d, "champion") ? "정지" : d.model_runtime?.champion?.memory_scope === "worker" ? "과거 구간 실행" : lag(health.lag_seconds),
  );

  text(
    "opCandidateLag",
    !modelIsRunning(d, "candidate") ? "정지" : d.model_runtime?.candidate?.memory_scope === "worker" ? "과거 구간 실행" : lag(candidate.lag_seconds),
  );

  const warnings = [];
  if (modelIsRunning(d, "champion") && health.status === "error")
    warnings.push(
      "Champion 판단 오류: " + (health.reason || "오류 기록 확인 필요"),
    );
  if (d.feed_metrics?.broker_error)
    warnings.push("시세 연결 오류: " + d.feed_metrics.broker_error);
  if (d.agent_process_running && m.agent_last_input_error)
    warnings.push("최근 입력 오류: " + m.agent_last_input_error);
  if (modelIsRunning(d, "candidate") && candidate.status === "error")
    warnings.push("Candidate 판단 오류: " + candidate.reason);
  else if (modelIsRunning(d, "candidate") && candidate.status === "stale")
    warnings.push(
      "Candidate 필수 관찰 처리가 feed보다 " +
        lag(candidate.lag_seconds) +
        " 뒤처져 있습니다. 미처리 관찰 " +
        whole(candidate.pending) +
        "개는 DB에 남아 있으며 아래 Candidate 손익도 마지막 처리 시각 기준입니다.",
    );
  if (modelIsRunning(d, "champion") && health.status === "stale")
    warnings.push(
      "Champion이 시세보다 " + lag(health.lag_seconds) + " 뒤처져 있습니다.",
    );
  const previousErrors = [];
  for (const role of ["champion", "candidate"]) {
    const error = m["last_" + role + "_error"];
    if (!error) continue;
    const name = role === "champion" ? "Champion" : "Candidate";
    if (modelIsRunning(d, role) && d.model_runtime?.[role]?.memory_scope !== "worker")
      warnings.push("최근 " + name + " 학습 오류: " + error);
    else previousErrors.push(name + " 지난 실행: " + error);
  }
  text("previousLearningErrors", previousErrors.join("\n") || "지난 학습 오류 없음");
  property("previousLearningErrorsPanel", "hidden", !previousErrors.length);
  property("runtimeAlertPanel", "hidden", !warnings.length);
  text(
    "runtimeAlertSummary",
    "최근 오류·지연 " + whole(warnings.length) + "건 · 원인 확인",
  );
  property("runtimeAlert", "hidden", !warnings.length);
  text("runtimeAlert", warnings.join(" "));
}

function renderObservationStatus(d) {
  const metrics = d.metrics || {},
    health = d.agent_health || {},
    candidate = health.candidate || {};
  const championDecision =
    metrics.champion_last_full_decision_timestamp ||
    d.account_observability?.champion?.policy?.timestamp;
  const candidateDecision =
    d.account_observability?.candidate?.last_full_decision_timestamp;
  text(
    "opChampionTime",
    "관찰 처리 " +
      timeOf(health.agent_cursor_timestamp_utc) +
      " · 실제 판단 " +
      timeOf(championDecision),
  );
  text(
    "opCandidateQueue",
    "처리 대기 " +
      whole(candidate.pending) +
      "개 · 관찰 처리 " +
      timeOf(candidate.cursor_timestamp_utc) +
      " · 실제 판단 " +
      timeOf(candidateDecision),
  );
  if (d.observe_enabled === false) {
    for (const role of runningModelRoles(d)) {
      badge(role + "LiveBadge", "실행 중 · 새 판단 중지", "");
      text(
        role === "champion" ? "opChampionLag" : "opCandidateLag",
        "판단 OFF",
      );
    }
    text("opChampionTime", modelIsRunning(d, "champion") ? "새 판단 중지 · 시세 수집 계속" : "모델 정지 · 마지막 기록 유지");
    text("opCandidateQueue", modelIsRunning(d, "candidate") ? "새 판단 중지 · 저장 경험 학습은 설정에 따름" : "모델 정지 · 미처리 경험 보존");
  } else {
    const m = d.metrics || {},
      observer = d.candidate_live_account || {};
    if (modelIsRunning(d, "champion") && m.champion_inference_skipped_reason === "context_only")
      badge("championLiveBadge", "문맥 갱신 · 풀 추론 없음", "");
    if (modelIsRunning(d, "candidate") && observer.status === "context_only")
      badge("candidateLiveBadge", "문맥 갱신 · 풀 추론 없음", "");
  }
  for (const role of runningModelRoles(d)) {
    const runtime = d.model_runtime?.[role];
    if (runtime?.memory_scope !== "worker") continue;
    text(role === "champion" ? "opChampionTime" : "opCandidateQueue", "ETHUSDT 데이터 시점 " + marketDataTime(runtime.last_decision));
    text(role === "champion" ? "opChampionLag" : "opCandidateLag", d.observe_enabled === false ? "판단 중지" : "과거 구간 실행");
  }

}

// Shared safe row builder; cached html() keeps table nodes on unchanged data.

// The overview owns one ledger scope. Never mix a MoE worker with legacy counters.
function overviewExperience(d) {
  const roles = moeModelRoles(d);
  if (roles.length && runningModelRoles(d).every((role) => modelUsesHistoricalData(d, role))) {
    const replays = roles.map((role) => d.model_runtime[role].replay || {});
    const sum = (key) => replays.every((r) => r[key] != null)
      ? replays.reduce((total, r) => total + num(r[key]), 0) : null;
    return {pending:sum("pending"), eligible:replays.every((r) => r.remaining_for_update != null) ? sum("remaining_for_update") : sum("untrained"), bytes:sum("bytes"),
      saving:sum("awaiting_checkpoint"),
      completed:replays.every((r) => Array.isArray(r.daily))
        ? replays.reduce((total,r) => total + r.daily.reduce((n,day) => n + num(day.completed),0),0) : null,
      held:sum("quarantined"), scope:"TradingMoE 전용 DB"};
  }
  const m = d.metrics || {};
  return {pending:m.replay_pending_count ?? m.pending_experiences, eligible:m.replay_eligible_backlog,
    completed:(m.daily_learning || []).reduce((n,r) => n + num(r.completed),0), bytes:m.replay_file_bytes,
    held:num(m.replay_quarantined_count)+num(m.replay_unsupported_count), scope:"기존 공통 DB"};
}

function operatorMetrics(d) {
  const m = d.metrics || {},
    v = d.validation_comparison || {};
  const known = (x) => x != null && Number.isFinite(Number(x));
  const count = (x) => (known(x) ? whole(x) : "—");
  const sumTrades = (books) =>
    ["KRW", "USD"].every((k) => known(books?.[k]?.trade_count))
      ? Number(books.KRW.trade_count) + Number(books.USD.trade_count)
      : null;
  const todayKey = new Intl.DateTimeFormat("sv-SE", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  const today = (m.daily_learning || []).find((row) => row.day === todayKey);
  const champion = sumTrades(d.paper_financials),
    candidate = sumTrades(d.candidate_live_account?.books);
  const ratio = (n, total) =>
    known(n) && known(total) && Number(total) > 0
      ? Math.max(0, Math.min(100, (100 * Number(n)) / Number(total)))
      : null;
  const state = learningSituation(d);
  const cards = {
    Paper: {
      number:
        champion == null || candidate == null
          ? "—"
          : whole(champion + candidate) + "건",
      status:
        d.paper_enabled == null
          ? "미확인"
          : d.paper_enabled
            ? "체결 허용"
            : "체결 중지",
      detail:
        "Champion " +
        count(champion) +
        "건 · Candidate " +
        count(candidate) +
        "건 · 계좌 시작부터 누적",
      ratio: null,
      tone: d.paper_enabled ? "good" : "",
    },
    Learning: {
      number: count(today?.completed) + " / " + count(today?.enqueued) + "건",
      status: state.title,
      detail:
        "남은 학습 " +
        count(m.replay_eligible_backlog) +
        "건 · 손익 확인 " +
        count(m.replay_pending_count ?? m.pending_experiences) +
        "건",
      ratio: ratio(today?.completed, today?.enqueued),
      tone: state.error
        ? "bad"
        : modelIsLearning(d, "champion") || modelIsLearning(d, "candidate")
          ? "blue"
          : "",
    },
    Trial: {
      number: count(v.bars_current) + " / " + count(v.bars_required) + "시점",
      status: !d.agent_process_running
        ? "정지"
        : v.active
          ? d.observe_enabled === false
            ? "중지"
            : "진행 중"
          : v.comparison_valid
            ? "판정 완료"
            : "시험 대기",
      detail: v.active
        ? "새 시세로 평가 · 판정 " +
          (d.daily_cycle?.next_reset_utc
            ? timeOf(d.daily_cycle.next_reset_utc)
            : "시각 미확인")
        : v.comparison_valid
          ? v.status === "promoted"
            ? "Candidate 승급"
            : "Champion 유지"
          : "새 고정 시험본 대기",
      ratio: ratio(v.bars_current, v.bars_required),
      tone: v.active ? "blue" : "",
    },
  };
  const runningRoles = moeModelRoles(d);
  if (runningRoles.length && runningModelRoles(d).every((role) => modelUsesHistoricalData(d,role))) {
    const experience = overviewExperience(d);
    Object.assign(cards.Learning, {number:count(experience.eligible)+"건 학습 대기",
      detail:"MoE DB · 저장 완료 "+count(experience.completed)+"건 · 학습 후 저장 대기 "+count(experience.saving)+"건 · 결과 대기 "+count(experience.pending)+"건", ratio:null});
    if (runningModelRoles(d).length) {
      const nativeTrades = runningRoles.map((role) => Object.values(d.model_runtime[role].books || {}).reduce((n,b) => n + num(b.trade_count),0));
      Object.assign(cards.Paper, {number:whole(nativeTrades.reduce((n,v) => n+v,0))+"건",
        detail:runningRoles.map((role,i) => (role === "champion" ? "Champion" : "Candidate")+" MoE "+whole(nativeTrades[i])+"건").join(" · ")});
    }
    Object.assign(cards.Trial, {number:"미실행",status:"기존 모델 승급전",detail:"MoE 가상매매와 별도 · 시험계좌 미실행",ratio:null,tone:""});
  }
  if (d.status_unavailable)
    for (const card of Object.values(cards))
      Object.assign(card, {
        number: "—",
        status: "연결 끊김",
        detail: "최신 수치를 확인할 수 없습니다.",
        ratio: null,
        tone: "bad",
      });
  return cards;
}

function renderWorkflowStatus(d) {
  const cards = operatorMetrics(d);
  renderOptimizerBoard(d);
  if (d.status_unavailable) {
    badge("gpuBadge", "연결 끊김", "bad");
    text("gpuMemory", "—");
    text("gpuPercent", "측정값 미확인");
    styleWidth("gpuFill", "0%");
    attribute("gpuMeter", "aria-valuenow", "0");
  }
  for (const [key, card] of Object.entries(cards)) {
    const id = "workflow" + key;
    text(id, card.number);
    badge(id + "State", card.status, card.tone);
    const detail = key === "Learning" ? "learningAtGlance" : id + "Detail";
    text(detail, card.detail);
    attribute(detail, "title", card.detail);
    if (key !== "Paper") {
      property(id + "Meter", "value", card.ratio ?? 0);
      attribute(
        id + "Meter",
        "aria-valuetext",
        card.ratio == null ? "미측정" : decimal(card.ratio, 1) + "%",
      );
      property(
        id + "Meter",
        "className",
        card.ratio == null ? "unmeasured" : "",
      );
    }
  }
}

function renderOverviewExperience(d) {
  const experience = overviewExperience(d);
  const counts = {
    Pending: experience.pending,
    Eligible: experience.eligible,
    Completed: experience.completed,
  };
  for (const [stage, count] of Object.entries(counts))
    text(
      "overviewExperience" + stage,
      count == null ? "—" : whole(count) + "건",
    );
  const situation = learningSituation(d);
  text("overviewLearningReason", situation.title);
}

function renderProcessHealth(d) {
  if (d.status_unavailable) {
    badge("processHealthBadge", "연결 끊김", "bad");
    text("processHealthValue", "상태 미확인");
    text("processHealthDetail", "서버 재연결 후 실행 상태 확인");
    text("overviewReplaySize", "—");
    return;
  }
  const parts = [
    ["웹서버", true],
    ["Feed", d.feed_running],
    ["Champion", modelIsRunning(d, "champion")],
    ["Candidate", modelIsRunning(d, "candidate")],
  ];
  const roles = runningModelRoles(d);
  const transitioning = Object.values(d.model_runtime || {}).some((item) => ["loading", "saving"].includes(item.status));
  const problem = Object.values(d.model_runtime || {}).some((item) => item.error || item.requested && ["stopped", "error"].includes(item.status));
  const ready = d.feed_running && roles.length > 0 && !problem && !transitioning;
  const historical = roles.length && roles.every((role) => modelUsesHistoricalData(d,role));
  badge(
    "processHealthBadge",
    problem ? "모델 오류" : transitioning ? "모델 전환 중" : ready ? "실행 중" : d.feed_running ? "시세 수집 중" : "정지",
    problem ? "bad" : transitioning ? "warn" : d.feed_running ? "good" : "",
  );
  text(
    "processHealthValue",
    problem ? "모델 오류 확인" : transitioning ? "모델 로딩·저장 중" : roles.length ? d.observe_enabled === false ? "모델 판단 중지" : historical ? "MoE 과거 가상매매" : "실시간 모델 실행" : d.feed_running ? "시세 수집 · 모델 정지" : "시스템 정지",
  );
  text(
    "processHealthDetail",
    parts.map(([name, on]) => name + " " + (on ? "실행" : "정지")).join(" · "),
  );
  const experience = overviewExperience(d), bytes = experience.bytes;
  text(
    "overviewReplaySize",
    bytes == null ? "—" : (Number(bytes) / 1048576).toFixed(1) + " MiB",
  );
  badge("overviewReplayBadge", bytes == null ? "미측정" : "DB 보존", "");
  text(
    "overviewReplayDetail",
    experience.scope + " · 학습 대기 " + whole(experience.eligible) + "건 · 보류 " + whole(experience.held) + "건",
  );
}

function renderOutputDiagnostics(d) {
  const x = d.output_diagnostics || {},
    message = x.warning
      ? "모델 출력 점검 경고: " +
        whole(x.largest_identical_group) +
        "종목의 확률 일치 (원인 조사 필요)"
      : "모델 출력 점검: " +
        whole(x.unique_probability_vectors) +
        " 종류 / " +
        whole(x.symbols_with_probabilities) +
        " 종목";
  text(
    "outputDiagnostic",
    message + " | 최신 판단 전체 매수는 매수 추천이 아닙니다. 실제 주문 OFF",
  );
}
