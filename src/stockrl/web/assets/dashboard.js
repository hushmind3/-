"use strict";

// Market overview, symbol decisions and parallel paper accounts.

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
  text(
    "modelRuntimeParameterCount",
    d.metrics?.parameters == null
      ? "모델 대기 중"
      : whole(d.metrics.parameters) + "개",
  );
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
  badge(
    "agentBadge",
    d.agent_running
      ? recent(m.last_market_timestamp)
        ? "추론 실행"
        : "모델 대기"
      : "모델 정지",
    d.agent_running
      ? recent(m.last_market_timestamp)
        ? "good"
        : "warn"
      : "bad",
  );
  text("decisionCount", whole(m.decisions) + "건");

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
    connected ? "소켓 연결" : "소켓 미연결",
    connected ? "good" : "warn",
  );
  text("brokerState", btitle);
  text("brokerDetail", bdetail);
  const used = num(m.cuda_memory_allocated_bytes),
    total = num(m.cuda_total_memory_bytes),
    ratio = total ? Math.min(100, (used / total) * 100) : 0,
    cuda = d.agent_running && String(m.device || "").startsWith("cuda"),
    mps = d.agent_running && String(m.device || "").startsWith("mps");
  badge(
    "gpuBadge",
    cuda
      ? "CUDA 적용"
      : mps
        ? "MPS 적용"
        : d.agent_running
          ? "CPU 사용"
          : "모델 정지",
    cuda || mps ? "good" : d.agent_running ? "warn" : "bad",
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
  text(
    "championVersion",
    "현재 champion · " + (d.champion_version || "확인 중"),
  );
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
  document.querySelectorAll("[data-market]").forEach(
    (el) =>
      (el.onclick = () => {
        selectedMarket = el.dataset.market;
        renderDecisions(d);
        $("decisions").scrollIntoView({ behavior: "smooth" });
      }),
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
  document.querySelectorAll("[data-filter]").forEach(
    (el) =>
      (el.onclick = () => {
        selectedMarket = el.dataset.filter;
        renderDecisions(d);
      }),
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
  }
}

function instrumentLabel(symbol, d) {
  const item = (d.instruments || []).find((row) => row.symbol === symbol);
  const configured = item?.name?.trim();
  return configured && !configured.includes("�")
    ? configured
    : SYMBOL_NAMES[symbol] || symbol;
}

function accountHoldings(items, d) {
  return ["KRW", "USD"]
    .map(
      (currency) =>
        currency +
          ": " +
          (items || [])
            .filter((p) => (p.currency || "KRW") === currency)
            .map(
              (p) =>
                instrumentLabel(p.symbol, d) + " " + whole(p.quantity) + "주",
            )
            .join(" · ") || "보유 없음",
    )
    .join(" | ");
}

function currencyPair(books, key) {
  return ["KRW", "USD"]
    .map(
      (currency) =>
        currency +
        " " +
        Number(books?.[currency]?.[key] || 0).toLocaleString("ko-KR", {
          maximumFractionDigits: 2,
        }),
    )
    .join(" · ");
}

function accountFills(fills, d) {
  return (
    (fills || [])
      .slice(-12)
      .reverse()
      .map(
        (x) =>
          instrumentLabel(x.symbol, d) +
          " " +
          x.action +
          " " +
          whole(x.quantity) +
          "주 @ " +
          Number(x.price || 0).toLocaleString("ko-KR", {
            maximumFractionDigits: 2,
          }),
      )
      .join(" · ") || "체결 없음"
  );
}

function renderValidationAccounts(d) {
  const v = d.validation_comparison || {};
  const books = v.accounts_available
    ? { champion: v.champion || {}, candidate: v.candidate || {} }
    : null;
  const krw = (name) => books?.[name]?.KRW || null,
    usd = (name) => books?.[name]?.USD || null;
  const money = (b, c) =>
    !b || b.net_pnl == null
      ? "?"
      : Number(b.net_pnl).toLocaleString("ko-KR", {
          maximumFractionDigits: 2,
        }) +
        " " +
        c +
        " (" +
        (Number(b.net_return_rate || 0) * 100).toFixed(2) +
        "%)";
  const split = (b) =>
    !b
      ? "?"
      : Number(b.realized_pnl || 0).toLocaleString("ko-KR", {
          maximumFractionDigits: 2,
        }) +
        " / " +
        Number(b.unrealized_pnl || 0).toLocaleString("ko-KR", {
          maximumFractionDigits: 2,
        });
  const costs = (b) =>
    !b
      ? "?"
      : Number(b.costs || 0).toLocaleString("ko-KR", {
          maximumFractionDigits: 2,
        });
  const counts = (a, b, key) =>
    !a || !b ? "?" : whole(a[key]) + " / " + whole(b[key]);
  const holdings = (b) =>
    !b
      ? "?"
      : (b.positions || [])
          .map(
            (p) =>
              instrumentLabel(p.symbol, d) + " " + whole(p.quantity) + "주",
          )
          .join(" · ") || "보유 없음";
  const pair = (name, key) =>
    books ? currencyPair(books[name] || {}, key) : "?";
  const directions = (name) => {
    const rows = (v.last_decisions?.[name] || []).map(
      (x) => instrumentLabel(x.symbol, d) + " " + x.action,
    );
    return rows.length
      ? rows.slice(0, 16).join(" · ") +
          (rows.length > 16 ? " · 외 " + (rows.length - 16) + "종목" : "")
      : "판정 대기";
  };
  text("trialChampionSeed", pair("champion", "initial_cash"));
  text("trialCandidateSeed", pair("candidate", "initial_cash"));
  text("trialChampionEquity", pair("champion", "equity"));
  text("trialCandidateEquity", pair("candidate", "equity"));
  text("trialChampionKrwPnl", money(krw("champion"), "KRW"));
  text("trialCandidateKrwPnl", money(krw("candidate"), "KRW"));
  text("trialChampionUsdPnl", money(usd("champion"), "USD"));
  text("trialCandidateUsdPnl", money(usd("candidate"), "USD"));
  text("trialChampionKrwSplit", split(krw("champion")) + " KRW");
  text("trialCandidateKrwSplit", split(krw("candidate")) + " KRW");
  text("trialChampionUsdSplit", split(usd("champion")) + " USD");
  text("trialCandidateUsdSplit", split(usd("candidate")) + " USD");
  text(
    "trialChampionCosts",
    costs(krw("champion")) + " KRW / " + costs(usd("champion")) + " USD",
  );
  text(
    "trialCandidateCosts",
    costs(krw("candidate")) + " KRW / " + costs(usd("candidate")) + " USD",
  );
  text(
    "trialChampionTrades",
    counts(krw("champion"), usd("champion"), "trade_count"),
  );
  text(
    "trialCandidateTrades",
    counts(krw("candidate"), usd("candidate"), "trade_count"),
  );
  text(
    "trialChampionPositions",
    counts(krw("champion"), usd("champion"), "position_count"),
  );
  text(
    "trialCandidatePositions",
    counts(krw("candidate"), usd("candidate"), "position_count"),
  );
  text("trialChampionDirections", directions("champion"));
  text("trialCandidateDirections", directions("candidate"));
  text("trialChampionHoldings", holdings(krw("champion")));
  text("trialCandidateHoldings", holdings(krw("candidate")));
  text("trialChampionUsdHoldings", holdings(usd("champion")));
  text("trialCandidateUsdHoldings", holdings(usd("candidate")));
  const bars = whole(v.bars_current),
    required = whole(v.bars_required || 390),
    status = v.status || "not_started";
  const progressed = bars > 0 || ["promoted", "rejected"].includes(status),
    trialPaused = !!v.active && d.observe_enabled === false;
  const invalid =
    status === "discarded" || (status === "rejected" && !v.comparison_valid);
  const check = (label, ok) =>
    label +
    ": " +
    (invalid
      ? "이전 시험 무효"
      : ok
        ? "일치"
        : progressed
          ? "불일치"
          : "대결 시작 대기");
  const fairness = [
    check("시작 자금", v.same_starting_cash),
    check("시장 입력", v.same_market_input),
    check("시장 봉", v.same_market_timeline && v.same_last_bar),
    check("비용 규칙", v.same_cost_rules),
    check("행동 선택 규칙", v.same_action_rule),
  ];
  fairness.push(
    trialPaused
      ? "승급전 일시정지 · 판단 OFF"
      : v.comparison_valid
        ? "하루 승급전 대결 유효"
        : v.active
          ? "대결 진행 중"
          : invalid
            ? "새 시험 대기"
            : "유효한 대결 결과 대기",
  );
  const reason =
    v.reason === "restart invalidated the in-memory validation snapshot"
      ? "서버 재시작으로 메모리 시험본이 사라져 무효 처리됨"
      : v.reason;
  text(
    "validationFairnessChecks",
    fairness.join(" · ") + (reason ? " · " + reason : ""),
  );
  let label = trialPaused
    ? "승급전 일시정지 · 판단 OFF · " + bars + " / " + required + "봉"
    : v.active
      ? "정식 대결 진행 중 · " + bars + " / " + required + "봉"
      : status === "promoted"
        ? "정식 대결 완료 · " + bars + " / " + required + "봉 · Candidate 승급"
        : status === "rejected" && v.comparison_valid
          ? "정식 대결 완료 · " + bars + " / " + required + "봉 · Champion 유지"
          : status === "rejected"
            ? "대결 무효 · " + bars + " / " + required + "봉 · 새 시험 대기"
            : status === "discarded"
              ? "이전 대결 무효 · " +
                bars +
                " / " +
                required +
                "봉 · 새 시험 대기"
              : status === "collecting"
                ? "시험 계좌 확인 중 · " + bars + " / " + required + "봉"
                : d.learning?.candidate_training
                  ? "Candidate 학습 중 · 학습 완료 후 대결 시작"
                  : "새 Candidate 대결 대기";
  text(
    "validationTrialStatus",
    label +
      (v.snapshot_version == null
        ? ""
        : " · 시험본 v" + whole(v.snapshot_version)) +
      (v.start_after ? " · 시작 기준 " + timeOf(v.start_after) : "") +
      (v.last_timestamp ? " · 마지막 봉 " + timeOf(v.last_timestamp) : ""),
  );
  badge(
    "validationTrialBadge",
    trialPaused
      ? "일시정지"
      : v.active
        ? "정식 대결"
        : status === "promoted"
          ? "Candidate 승급"
          : status === "rejected" && v.comparison_valid
            ? "Champion 유지"
            : status === "rejected"
              ? "재시험 필요"
              : "대기",
    trialPaused
      ? "warn"
      : v.active
        ? "blue"
        : status === "rejected" && !v.comparison_valid
          ? "bad"
          : "",
  );
}

function renderCandidateLiveAccount(d) {
  const v = d.candidate_live_account || {},
    books = v.books || {},
    krw = books.KRW,
    usd = books.USD,
    money = (b) =>
      !b
        ? "?"
        : Number(b.net_pnl || 0).toLocaleString("ko-KR", {
            maximumFractionDigits: 2,
          }) +
          " (" +
          (Number(b.net_return_rate || 0) * 100).toFixed(2) +
          "%)",
    split = (b) =>
      !b
        ? "?"
        : Number(b.realized_pnl || 0).toLocaleString("ko-KR", {
            maximumFractionDigits: 2,
          }) +
          " / " +
          Number(b.unrealized_pnl || 0).toLocaleString("ko-KR", {
            maximumFractionDigits: 2,
          }),
    positions = ["KRW", "USD"].flatMap((currency) =>
      (books[currency]?.positions || []).map((p) => ({ ...p, currency })),
    ),
    decisions =
      (v.decisions || [])
        .map((x) => instrumentLabel(x.symbol, d) + " " + x.action)
        .join(" · ") || "최근 판단 없음",
    status = !d.agent_process_running
      ? "정지 · 마지막 관찰 " +
        (v.last_observation_timestamp
          ? timeOf(v.last_observation_timestamp)
          : "기록 없음")
      : d.observe_enabled === false
        ? "판단 OFF · 기존 계좌 평가 유지"
        : v.status === "judgment_paused"
          ? "새 판단 중지 · 계좌 평가 유지"
          : v.status === "context_only"
            ? "문맥 갱신 · 풀 추론 없음"
            : v.candidate_training
              ? "학습 중 · 최근 완료 가중치 v" + whole(v.candidate_version)
              : v.status === "observing"
                ? "실시간 관찰 중 · v" + whole(v.candidate_version)
                : v.status === "training_and_observing"
                  ? "학습 병행 · 최근 완료 버전 v" + whole(v.candidate_version)
                  : v.status === "error"
                    ? "관찰 오류: " + (v.error || "확인 필요")
                    : "Candidate 학습 업데이트 대기";
  text("candidateLiveStatus", status);
  text("candidateLiveSeed", currencyPair(books, "initial_cash"));
  text("candidateLiveCash", currencyPair(books, "cash"));
  text("candidateLiveKrwNet", money(krw) + " KRW");
  text("candidateLiveUsdNet", money(usd) + " USD");
  text("candidateLiveKrwSplit", split(krw) + " KRW");
  text("candidateLiveUsdSplit", split(usd) + " USD");
  text(
    "candidateLiveCosts",
    Number(krw?.costs || 0).toLocaleString("ko-KR", {
      maximumFractionDigits: 2,
    }) +
      " KRW / " +
      Number(usd?.costs || 0).toLocaleString("ko-KR", {
        maximumFractionDigits: 2,
      }) +
      " USD",
  );
  text(
    "candidateLiveTrades",
    whole(krw?.trade_count) + " KRW · " + whole(usd?.trade_count) + " USD",
  );
  text(
    "candidateLivePositionCount",
    whole(krw?.position_count) +
      " KRW · " +
      whole(usd?.position_count) +
      " USD",
  );
  text("candidateLiveHoldings", accountHoldings(positions, d));
  text("candidateLiveDirections", decisions);
  text("candidateLiveFills", accountFills(v.recent_fills, d));
  text(
    "candidateLiveLastInference",
    "실제 모델 판단 " +
      (v.last_full_decision_timestamp
        ? timeOf(v.last_full_decision_timestamp)
        : "기록 없음") +
      " · 계좌 평가 " +
      (v.last_observation_timestamp
        ? timeOf(v.last_observation_timestamp)
        : "미측정") +
      (v.last_inference_seconds == null
        ? ""
        : " · 추론 " + Number(v.last_inference_seconds).toFixed(2) + "초"),
  );
  badge(
    "candidateLiveBadge",
    v.status === "error" ? "오류" : v.available ? "paper 관찰" : "대기",
    v.status === "error" ? "bad" : "blue",
  );
}

function renderLiveAccountComparison(d) {
  const c = d.live_account_comparison || {},
    shared = c.shared_observation || {},
    lastBars = c.last_bar_timestamps_equal
      ? "마지막 관찰 시각 동일"
      : "Candidate 관찰 처리 대기";
  text(
    "liveAccountObservationNotice",
    "공통 시장 입력을 양쪽 모두 필수로 받으며 같은 체결·비용·난수 규칙을 사용합니다. " +
      lastBars +
      " · 미처리 관찰 " +
      whole(shared.pending) +
      "개는 같은 DB에 보존합니다. 각자의 행동·손익 경험을 합쳐 둘 다 학습합니다. 옛 방식에서 건너뛴 " +
      whole(c.candidate_skipped_observations) +
      "개는 과거 수치입니다. 계속 학습 중인 운영 계좌 누적 성적과 고정 시험본의 하루 승급전 점수는 구분합니다.",
  );
}

function renderLiveAccounts(d) {
  const pf = d.paper_financials || {},
    books = { KRW: pf.KRW || {}, USD: pf.USD || {} },
    positions = Object.values(d.paper_positions || {});
  text("championLiveSeed", currencyPair(books, "initial_cash"));
  text("championLiveCash", currencyPair(books, "cash"));

  text(
    "championLivePositionCount",
    whole(books.KRW.position_count) +
      " KRW · " +
      whole(books.USD.position_count) +
      " USD",
  );
  text("positions", accountHoldings(positions, d));
  text(
    "modelDirections",
    Object.entries(d.model_directions || {})
      .map(
        ([symbol, direction]) =>
          instrumentLabel(symbol, d) +
          " " +
          (Number(direction) > 0
            ? "매수"
            : Number(direction) < 0
              ? "매도"
              : "관망"),
      )
      .join(" | ") || "없음",
  );
  text("championLiveFills", accountFills(d.paper_account?.fills, d));
  renderValidationAccounts(d);
  renderCandidateLiveAccount(d);
  renderLiveAccountComparison(d);
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

function accountMoney(value, currency) {
  return value == null
    ? "—"
    : Number(value).toLocaleString("ko-KR", {
        maximumFractionDigits: currency === "KRW" ? 0 : 2,
      }) +
        " " +
        currency;
}

function accountPercent(value) {
  return value == null ? "—" : (Number(value) * 100).toFixed(2) + "%";
}

function accountTable(headers, rows) {
  return (
    "<table><thead><tr>" +
    headers.map((x) => "<th>" + esc(x) + "</th>").join("") +
    "</tr></thead><tbody>" +
    rows
      .map(
        (row) =>
          "<tr>" + row.map((x) => "<td>" + esc(x) + "</td>").join("") + "</tr>",
      )
      .join("") +
    "</tbody></table>"
  );
}

function renderAccountDiagnostics(d) {
  const view = d.account_observability;
  if (!view) {
    for (const role of ["champion", "candidate"])
      text(role + "AccountOverview", "신규 계좌 지표 연결 대기");
    return;
  }
  for (const role of ["champion", "candidate"]) {
    const account = view[role] || {},
      books = account.books || {},
      currentPolicy = account.policy || {},
      isChampion = role === "champion";
    const health = isChampion ? d.agent_health : d.agent_health?.candidate;
    const active = Boolean(d.agent_process_running),
      enabled = Boolean(d.paper_enabled);
    const badge = isChampion ? "championLiveBadge" : "candidateLiveBadge";
    const error = account.observer_error || health?.status === "error";
    text(
      badge,
      error
        ? "판단 오류"
        : !active
          ? "정지 · 마지막 기록"
          : d.observe_enabled === false
            ? "판단 OFF · 계좌 평가 유지"
            : health?.status === "stale"
              ? "시세보다 " +
                Math.ceil(num(health.lag_seconds) / 60) +
                "분 지연"
              : enabled
                ? "판단·가상매매"
                : "관찰만",
    );
    property(
      badge,
      "className",
      "pill " +
        (error
          ? "danger"
          : health?.status === "stale"
            ? "warn"
            : active && health?.status === "healthy"
              ? "good"
              : ""),
    );
    const currentHasTrades = num(currentPolicy.observed_tradable_symbols) > 0;
    const policy = currentHasTrades
      ? currentPolicy
      : account.last_tradable_policy || {};
    html(role + "AccountOverview", () =>
      ["KRW", "USD"]
        .map((currency) => {
          const b = books[currency];
          if (!b)
            return (
              '<div class="account-currency">' + currency + " · 계좌 대기</div>"
            );
          const tone =
            num(b.net_pnl) < 0 ? "account-negative" : "account-positive";
          return (
            '<div class="account-currency"><div class="currency-title">' +
            currency +
            ' · 비용 차감 수익률</div><strong class="' +
            tone +
            '">' +
            esc(accountPercent(b.net_return_rate)) +
            '</strong><div class="net-value ' +
            tone +
            '">' +
            esc(accountMoney(b.net_pnl, currency)) +
            "</div><small>현재 순자산 " +
            esc(accountMoney(b.equity, currency)) +
            "<br>현금 " +
            esc(accountPercent(b.cash_ratio)) +
            " · 최대 종목 비중 " +
            esc(accountPercent(b.largest_position_weight)) +
            "<br>" +
            (!num(b.trade_count)
              ? "아직 체결 없음"
              : "누적 " + whole(b.trade_count) + "건 체결") +
            "</small></div>"
          );
        })
        .join(""),
    );
    const pair = (key) =>
      ["KRW", "USD"].map((c) => accountMoney(books[c]?.[key], c)).join(" · ");
    text(
      isChampion ? "championLiveSeed" : "candidateLiveSeed",
      pair("initial_cash"),
    );
    text(isChampion ? "championLiveCash" : "candidateLiveCash", pair("cash"));
    text(isChampion ? "paperCosts" : "candidateLiveCosts", pair("costs"));
    text(
      isChampion ? "paperTradeCount" : "candidateLiveTrades",
      ["KRW", "USD"]
        .map((c) => c + " " + whole(books[c]?.trade_count) + "건")
        .join(" · "),
    );
    text(
      isChampion ? "championLivePositionCount" : "candidateLivePositionCount",
      ["KRW", "USD"]
        .map((c) => c + " " + whole(books[c]?.position_count) + "종목")
        .join(" · "),
    );
    const readings = [];
    for (const currency of ["KRW", "USD"]) {
      const b = books[currency];
      if (!b) continue;
      if (!num(b.trade_count)) {
        readings.push(
          currency + "은 체결이 없어 아직 매매 성과를 평가할 수 없습니다.",
        );
        continue;
      }
      readings.push(
        currency +
          " 순손익 " +
          accountMoney(b.net_pnl, currency) +
          ", 누적 비용 " +
          accountMoney(b.costs, currency) +
          ". 기록된 체결 손익에 비용만 더하면 " +
          accountMoney(b.recorded_pnl_plus_costs, currency) +
          "입니다.",
      );
      if (!b.reconciliation_ok)
        readings.push(
          currency +
            " 손익 합산 불일치: " +
            accountMoney(b.reconciliation_difference, currency),
        );
    }
    text(
      role + "AccountReading",
      readings.join(" ") + " 비용을 더한 값은 무비용 재실험 결과가 아닙니다.",
    );
    const status = account.observer_error
      ? "관찰 오류: " + account.observer_error
      : account.training
        ? "학습 중 · 저장된 최신 가중치로 관찰"
        : "저장된 가중치로 관찰";
    const profile = account.inference_profile || mInferenceProfile(d, role),
      queue = isChampion ? 0 : num(d.metrics?.shared_observation?.pending);
    const part = (v) => (v == null ? "미측정" : decimal(v, 2) + "초");
    property(
      role + "AccountMeta",
      "className",
      "account-meta" + (health?.status === "stale" ? " delayed" : ""),
    );
    text(
      role + "AccountMeta",
      "마지막 실제 모델 판단 " +
        timeOf(account.last_full_decision_timestamp) +
        " · 계좌 평가·관찰 처리 " +
        timeOf(account.last_timestamp) +
        " · 시세 대비 " +
        (health?.lag_seconds == null
          ? "지연 미측정"
          : whole(health.lag_seconds) + "초 지연") +
        " · 관측 가중치 v" +
        whole(account.version) +
        ". 최근 GPU 대기 " +
        part(profile.wait_seconds) +
        " / 입력·계산 " +
        part(profile.forward_seconds) +
        (isChampion ? "" : " · 미처리 관찰 " + whole(queue) + "개"),
    );
    const positions = ["KRW", "USD"].flatMap((c) =>
      (books[c]?.positions || []).map((p) => [
        instrumentLabel(p.symbol, d),
        c,
        whole(p.quantity) + "주",
        accountMoney(p.average_cost, c),
        accountMoney(p.mark, c) + (p.mark_available ? "" : " (평단 대체)"),
        accountPercent(p.weight),
        accountMoney(p.unrealized_pnl, c),
      ]),
    );
    html(isChampion ? "positions" : "candidateLiveHoldings", () =>
      positions.length
        ? accountTable(
            [
              "종목",
              "통화",
              "수량",
              "평단",
              "평가가격",
              "계좌 비중",
              "평가손익",
            ],
            positions,
          )
        : "보유 종목 없음",
    );
    const costs = ["KRW", "USD"]
      .filter((c) => books[c])
      .map((c) => {
        const b = books[c];
        return [
          c,
          accountMoney(b.fees, c),
          accountMoney(b.sell_tax, c),
          accountMoney(b.spread, c),
          accountMoney(b.slippage, c),
          accountPercent(b.cost_return_rate),
          b.reconciliation_ok ? "일치" : "불일치",
        ];
      });
    html(role + "CostDetails", () =>
      accountTable(
        [
          "통화",
          "수수료",
          "세금",
          "스프레드",
          "슬리피지",
          "시드 대비 비용",
          "순손익 = 실현 + 평가",
        ],
        costs,
      ),
    );
    html(
      role + "TradingStatistics",
      () =>
        accountTable(
          [
            "통화",
            "통계 시작",
            "기록 매수 / 매도",
            "자연시간당 체결",
            "매도 체결 승률",
            "평균 매도 순손익",
            "평균 / 최단 / 최장 보유",
          ],
          ["KRW", "USD"]
            .filter((c) => books[c])
            .map((c) => {
              const stats = books[c].trade_statistics || {},
                seconds = (value) =>
                  value == null
                    ? "미측정"
                    : decimal(Number(value) / 60, 1) + "분";
              return [
                c,
                stats.first_timestamp
                  ? timeOf(stats.first_timestamp)
                  : "새 체결 대기",
                whole(stats.buy_count) + " / " + whole(stats.sell_count),
                decimal(stats.fills_per_elapsed_hour, 1),
                accountPercent(stats.sell_win_rate),
                accountMoney(stats.mean_sell_net_pnl, c),
                [
                  stats.mean_holding_seconds,
                  stats.holding_seconds_min,
                  stats.holding_seconds_max,
                ]
                  .map(seconds)
                  .join(" / "),
              ];
            }),
        ) +
        '<p class="help">통계 기록을 시작한 이후의 체결만 집계합니다. 승률은 비용을 뺀 매도 체결별 손익 기준이며 부분매도도 한 건입니다. 보유시간은 최초 진입에서 해당 매도까지이고, 과거 진입시각이 없는 보유분은 시간 집계에서 제외합니다.</p>',
    );
    const decisions = policy.decisions || [];
    const counts = policy.action_counts || {};
    const policyAsOf = currentHasTrades
      ? "이번 관측 "
      : "지금 새 거래 가능 시세는 0종목입니다. 마지막 거래 가능 관측 기록 ";
    text(
      role + "PolicySummary",
      decisions.length
        ? policyAsOf +
            timeOf(policy.timestamp) +
            " · " +
            whole(policy.observed_tradable_symbols) +
            "종목: 매수 " +
            whole(counts.BUY) +
            " / 관망 " +
            whole(counts.HOLD) +
            " / 매도 " +
            whole(counts.SELL) +
            " · 새 주문 " +
            whole(policy.submitted_orders) +
            "건 · 미보유 매도 " +
            whole(policy.sell_without_position) +
            "건. 평균 최고 확률 " +
            accountPercent(policy.mean_top_probability) +
            ", 확률 퍼짐 " +
            accountPercent(policy.mean_normalized_entropy) +
            " (100%는 세 행동의 확률이 같음). 최고 확률과 다른 추첨 " +
            whole(policy.sampled_actions_different_from_argmax) +
            "건. 현금 목표 KRW " +
            accountPercent(policy.effective_cash_target?.KRW) +
            " / USD " +
            accountPercent(policy.effective_cash_target?.USD) +
            ". 목표 비중은 주문 계산에 쓰는 값이며 현재 보유 비중과 다릅니다."
        : "새 거래 가능 시세의 정책 지표 수집 대기 · 오래된 판단을 현재 값으로 대신 표시하지 않습니다.",
    );
    html(isChampion ? "modelDirections" : "candidateLiveDirections", () =>
      decisions.length
        ? accountTable(
            [
              "종목",
              "판단",
              "매도 확률",
              "관망 확률",
              "매수 확률",
              "목표 비중",
              "보유량",
              "새 주문",
            ],
            decisions.map((row) => [
              instrumentLabel(row.symbol, d),
              { BUY: "매수", HOLD: "관망", SELL: "매도" }[row.action] ||
                row.action,
              accountPercent(row.p_sell),
              accountPercent(row.p_hold),
              accountPercent(row.p_buy),
              accountPercent(row.allocation_target),
              whole(row.held_quantity) + "주",
              row.order_submitted ? "제출" : "없음",
            ]),
          )
        : "관측 대기",
    );
    const fills = (account.recent_fills || []).slice().reverse();
    html(isChampion ? "championLiveFills" : "candidateLiveFills", () =>
      fills.length
        ? accountTable(
            ["시각", "종목", "체결", "수량", "체결가", "수수료", "매도세"],
            fills.map((f) => [
              timeOf(f.date),
              instrumentLabel(f.symbol, d),
              f.action === "BUY" ? "매수" : "매도",
              whole(f.quantity) + "주",
              accountMoney(f.price, f.currency),
              accountMoney(f.fee, f.currency),
              accountMoney(f.sell_tax, f.currency),
            ]),
          )
        : "체결 없음",
    );
  }
  text(
    "accountInputScope",
    "1·3·5·15·60분봉 + 일·주·월봉 요약을 함께 입력합니다. 저장된 마지막 시세 중 최우선 매수·매도 호가가 있는 종목 " +
      whole(view.quoted_bid_ask_symbols) +
      "개 (현재 수신 여부는 시장 수신 상태 참조). 전체 호가 단계는 미수집입니다. USD 비용 가정: 편도 수수료 " +
      accountPercent(view.fee_rate) +
      ", 슬리피지 " +
      decimal(view.slippage_bps, 1) +
      "bp. 왕복 비용은 약 " +
      accountPercent(view.usd_round_trip_cost_rate_before_spread) +
      "부터이며 스프레드는 별도입니다. 한국 매도세는 " +
      accountPercent(view.krw_sell_tax_assumption) +
      "의 모의 가정입니다. 실제 주문 OFF.",
  );
  text(
    "accountRewardHorizon",
    "현재 새 학습 경험은 " +
      num(d.metrics?.reward_credit?.duration_seconds / 60).toFixed(0) +
      "분 동안 결과를 연결하고 다음 상태의 예상 가치도 사용합니다. 시간봉 입력 길이와 별개입니다. 기존 짧은 경험은 이전 방식 그대로 학습하며, 미보유 매도·미체결 판단을 체결로 세지 않습니다.",
  );
}

function mInferenceProfile(d, role) {
  return d.metrics?.[role + "_live_inference_profile"] || {};
}

function renderOperationsOverview(d) {
  const m = d.metrics || {},
    health = d.agent_health || {},
    candidate = health.candidate || {},
    schedule = m.gpu_scheduler || {};
  const names = {
    champion_live: "Champion 판단",
    candidate_live: "Candidate 판단",
    champion_learning_step: "Champion 학습",
    candidate_learning_step: "Candidate 학습",
    champion_learning_setup: "Champion 학습 준비",
    candidate_learning_setup: "Candidate 학습 준비",
    candidate_publish: "Candidate 완료 가중치 반영",
    validation_champion: "Champion 승급전",
    validation_candidate: "Candidate 승급전",
  };
  const roleName = (role) => names[role] || role || "작업 대기",
    lag = (v) => (v == null ? "미측정" : whole(v) + "초");
  text(
    "opChampionLag",
    !d.agent_process_running ? "정지" : lag(health.lag_seconds),
  );

  text(
    "opCandidateLag",
    !d.agent_process_running ? "정지" : lag(candidate.lag_seconds),
  );

  text(
    "opGpuWork",
    !d.agent_process_running
      ? "정지"
      : schedule.policy
        ? roleName(schedule.active)
        : "새 계측 적용 대기",
  );
  const learningFirst = schedule.policy === "preopen_replay_learning_first";
  text(
    "opGpuQueue",
    schedule.policy
      ? (learningFirst
          ? "본장 전 학습 우선 · " +
            timeOf(schedule.learning_priority_until_utc) +
            "까지"
          : "실시간 판단 우선") +
          " · 대기 " +
          ((schedule.waiting || []).map((x) => roleName(x.role)).join(" → ") ||
            "없음")
      : "현재 실행 프로세스의 순서 계측 없음",
  );
  text("opLearningQueue", whole(m.replay_eligible_backlog) + "개");
  text("opLastLearning", "마지막 학습·저장 " + timeOf(m.last_update_utc));
  const learningOff = d.learning_enabled === false,
    waitReason = String(m.learning_wait_reason || "");
  if (learningOff) {
    const remaining = whole(m.replay_eligible_backlog || 0);
    text(
      "dualLearningStatus",
      "replay 학습 중지 · 미학습 " + remaining + "건 보존",
    );
    text(
      "learningFlowHealth",
      "사용자가 replay 학습을 중지했습니다. 관찰과 가상계좌는 각 설정대로 동작하고 미학습 경험은 DB에 남습니다.",
    );
    badge("learningState", "학습 OFF · 경험 보존", "");
  } else if (waitReason) {
    const schedulerWait = waitReason.includes("실시간 GPU 추론 요청");
    const label = schedulerWait ? "GPU가 실시간 판단 처리 중" : waitReason;
    text("dualLearningStatus", label + " · replay 경험 보존");
    text(
      "learningFlowHealth",
      waitReason +
        " · 학습 데이터는 삭제하지 않고 다음 학습 회차에 처리합니다.",
    );
    badge(
      "learningState",
      schedulerWait ? "GPU 추론 처리 중" : "학습 적용 대기",
      "warn",
    );
  }
  if (learningFirst && !learningOff) {
    text(
      "dualLearningStatus",
      "본장 전 replay 학습 우선 · 시세·미처리 관찰 보존",
    );
    text(
      "learningFlowHealth",
      "20시 이후에는 두 모델의 미학습 경험을 우선 처리합니다. 판단·승급전은 남는 GPU 시간을 사용하며 본장 시작부터 판단 우선으로 복귀합니다.",
    );
    badge("learningState", "본장 전 학습 우선", "blue");
  }
  const warnings = [];
  if (candidate.status === "error")
    warnings.push("Candidate 판단 오류: " + candidate.reason);
  else if (candidate.status === "stale")
    warnings.push(
      "Candidate 필수 관찰 처리가 feed보다 " +
        lag(candidate.lag_seconds) +
        " 뒤처져 있습니다. 미처리 관찰 " +
        whole(candidate.pending) +
        "개는 DB에 남아 있으며 아래 Candidate 손익도 마지막 처리 시각 기준입니다.",
    );
  if (health.status === "stale")
    warnings.push(
      "Champion이 시세보다 " + lag(health.lag_seconds) + " 뒤처져 있습니다.",
    );
  if (m.last_candidate_error)
    warnings.push("최근 Candidate 학습 오류: " + m.last_candidate_error);
  if (m.last_champion_error)
    warnings.push("최근 Champion 학습 오류: " + m.last_champion_error);
  $("runtimeAlert").hidden = !warnings.length;
  text("runtimeAlert", warnings.join(" "));
  // Use the same backend health and timestamps everywhere, rather than a
  // second client threshold based on feed-process wall-clock activity.
  if (d.agent_process_running) {
    badge(
      "agentBadge",
      health.status === "stale"
        ? "Champion 판단 지연"
        : health.status === "healthy"
          ? "Champion 판단 정상"
          : "판단 상태 확인",
      health.status === "stale" ? "warn" : "good",
    );
    text(
      "agentDetail",
      "시세 " +
        timeOf(health.latest_feed_timestamp_utc) +
        " / 완료 판단 " +
        timeOf(health.agent_cursor_timestamp_utc) +
        " · " +
        lag(health.lag_seconds) +
        " 지연",
    );
  }
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
  text(
    "agentDetail",
    "시세 " +
      timeOf(health.latest_feed_timestamp_utc) +
      " / 관찰 처리 " +
      timeOf(health.agent_cursor_timestamp_utc) +
      " / 실제 모델 판단 " +
      timeOf(championDecision),
  );
  if (d.agent_process_running && health.status === "healthy")
    badge("agentBadge", "관찰 처리 정상", "good");
  if (d.observe_enabled === false) {
    badge("agentBadge", "판단 OFF", "");
    text(
      "agentDetail",
      "사용자가 새 모델 판단을 중지했습니다. 시세 저장·보유 평가·replay 학습은 계속됩니다.",
    );
    for (const role of ["champion", "candidate"]) {
      badge(role + "LiveBadge", "판단 OFF", "");
      text(
        role === "champion" ? "opChampionLag" : "opCandidateLag",
        "판단 OFF",
      );
    }
    text("opChampionTime", "추론 중지 · 시세 저장·보유 평가 계속");
    text("opCandidateQueue", "추론 중지 · 기존 경험 처리·학습 계속");
  } else {
    const m = d.metrics || {},
      observer = d.candidate_live_account || {};
    if (m.champion_inference_skipped_reason === "context_only")
      badge("championLiveBadge", "문맥 갱신 · 풀 추론 없음", "");
    if (observer.status === "context_only")
      badge("candidateLiveBadge", "문맥 갱신 · 풀 추론 없음", "");
  }
}

// Shared safe row builder; cached html() keeps table nodes on unchanged data.
function tableRows(rows) {
  return rows
    .map(
      (row) =>
        "<tr>" +
        row.map((value) => "<td>" + esc(value) + "</td>").join("") +
        "</tr>",
    )
    .join("");
}

// Learning lifecycle counts use the same replay metrics as the worker.
function renderExperienceFlow(d) {
  const m = d.metrics || {};
  const pending = m.replay_pending_count ?? m.pending_experiences;
  const eligible = m.replay_eligible_backlog;
  const completed = (m.daily_learning || []).reduce(
    (sum, day) => sum + num(day.completed),
    0,
  );
  text("flowPending", pending == null ? "미측정" : whole(pending) + "건");
  text("flowEligible", eligible == null ? "미측정" : whole(eligible) + "건");
  text("flowCompleted", whole(completed) + "건");
  const pendingState = pendingOutcomeState(d);
  text("flowPendingWindow", pendingState.window);
  text("flowPendingWaitReason", pendingState.reason);
  const labels = {
    missing_market_input: "시세 입력 복구 필요",
    next_quote: "해당 종목의 후속 시세 대기",
    reward_horizon: "결과 평가 시간 경과 대기",
    fill: "실제 가상체결 대기",
    blocked: "입력·보상 형식 확인 필요",
  };
  const reasons = ["champion", "candidate"].map((role) => {
    const status = m[role + "_pending_reward_status"];
    if (!status)
      return (
        (role === "champion" ? "Champion" : "Candidate") + ": 평가 사유 미측정"
      );
    return (
      (role === "champion" ? "Champion" : "Candidate") +
      ": " +
      (Object.entries(status.reasons || {})
        .map(([key, count]) => (labels[key] || key) + " " + whole(count) + "건")
        .join(" · ") || "손익 확인 대기 없음")
    );
  });
  const settled =
    num(m.champion_closed_market_rewards_settled) +
    num(m.candidate_closed_market_rewards_settled);
  text(
    "pendingRewardReasons",
    reasons.join(" | ") +
      (settled
        ? " | 장 종료 후 실제 마지막 입력으로 평가 완료 " +
          whole(settled) +
          "건"
        : ""),
  );
  renderLearningSituation(d);
}

// Explain the same measured state everywhere; no inference or control requests.
function learningSituation(d) {
  if (d.status_unavailable)
    return {
      title: "서버 연결 끊김 · 학습 상태 미확인",
      reason: "최신 상태를 받을 수 없어 현재 학습 여부를 확인할 수 없습니다.",
      next: "5초마다 연결을 다시 확인합니다. 마지막 완료 내역은 이전에 확인된 기록입니다.",
      tone: "warn",
    };
  const m = d.metrics || {};
  const known = (v) => v != null && Number.isFinite(Number(v));
  const eligible = known(m.replay_eligible_backlog)
    ? Number(m.replay_eligible_backlog)
    : null;
  const pendingValue = m.replay_pending_count ?? m.pending_experiences;
  const pending = known(pendingValue) ? Number(pendingValue) : null;
  const error =
    m.learner_statistics_error ||
    m.agent_last_input_error ||
    m.last_champion_error ||
    m.last_candidate_error;
  const active = ["champion", "candidate"]
    .filter((role) => m[role + "_training"])
    .map((role) => (role === "champion" ? "Champion" : "Candidate"))
    .join(" · ");
  const wait = String(m.learning_wait_reason || "");
  if (d.agent_process_running !== true)
    return {
      title:
        d.agent_process_running === false
          ? "학습 프로세스 정지"
          : "학습 프로세스 상태 미확인",
      reason: "현재 실행 중인 학습 프로세스가 확인되지 않습니다.",
      next: "운영 · 계좌에서 시스템 실행 상태를 확인하세요.",
      tone: "warn",
      error,
    };
  if (d.learning_enabled === false)
    return {
      title: "학습 OFF · 경험 보존",
      reason: "사용자가 학습을 껐습니다. 미학습 경험은 DB에 남아 있습니다.",
      next: "운영 · 계좌에서 replay 학습을 ON으로 바꾸면 이어서 학습합니다.",
      tone: "",
      error,
    };
  if (error)
    return {
      title: "학습 오류 확인 필요",
      reason: error,
      next: "오류 해결이 필요합니다. 학습 완료로 표시하지 않습니다.",
      tone: "bad",
      error,
    };
  if (d.learning_enabled !== true)
    return {
      title: "학습 설정 미확인",
      reason: "서버가 학습 ON/OFF를 아직 보내지 않았습니다.",
      next: "다음 상태 수신에서 설정을 확인합니다.",
      tone: "warn",
    };
  if (active)
    return {
      title: active + (wait ? " 학습 회차 진행 · 연산 대기" : " 학습 중"),
      reason:
        wait || "replay 경험으로 가중치를 갱신하고 학습 결과를 저장합니다.",
      next: wait
        ? "대기 조건이 해소되면 현재 회차를 이어갑니다."
        : "두 모델의 학습·저장 완료를 확인한 경험부터 삭제합니다.",
      tone: "blue",
    };
  if (eligible === null || pending === null)
    return {
      title: "학습량 미확인",
      reason:
        "학습 가능한 경험과 결과 평가 대기 건수를 아직 확인하지 못했습니다.",
      next: "다음 상태 수신에서 실제 잔여량을 확인합니다.",
      tone: "warn",
    };
  const blocked =
    num(m.replay_quarantined_count) + num(m.replay_unsupported_count);
  if (eligible === 0) {
    const completed =
      ["champion", "candidate"].some(
        (role) => m[role + "_last_completed_round"]?.completed_utc,
      ) || (m.daily_learning || []).some((day) => num(day.completed) > 0);
    const title = blocked
      ? "학습 대기 · 보류 확인 필요"
      : completed
        ? pending
          ? "준비된 경험 모두 학습 완료 · 결과 대기"
          : "준비된 경험 모두 학습 완료 · 새 경험 대기"
        : pending
          ? "학습 ON · 손익 확인 중"
          : "학습 ON · 새 경험 대기";
    return {
      title,
      reason:
        "학습은 ON입니다. 지금 학습 가능한 경험은 0건이며, 손익 확인 중인 판단 기록은 " +
        whole(pending) +
        "건입니다." +
        (blocked
          ? " 학습할 수 없는 보류 " +
            whole(blocked) +
            "건은 별도 확인이 필요합니다."
          : ""),
      next: pending
        ? "손익이 확정된 기록부터 자동으로 학습합니다."
        : "새 판단의 손익이 확정되면 자동으로 학습합니다.",
      tone: blocked ? "warn" : "",
    };
  }
  return {
    title: wait ? "학습 ON · 처리 순서 대기" : "학습 ON · 다음 회차 준비",
    reason:
      (wait || "다음 학습 회차를 준비하고 있습니다.") +
      " 학습 가능한 경험 " +
      whole(eligible) +
      "건이 남아 있습니다.",
    next: wait
      ? "대기 조건이 해소되면 남은 경험부터 학습합니다."
      : "저장된 경험을 순서대로 학습합니다.",
    tone: "blue",
  };
}

function renderLearningSituation(d) {
  const m = d.metrics || {},
    state = learningSituation(d);
  badge("learningState", state.title, state.tone);
  text("learningAtGlance", state.title + " · 학습 상태 보기");
  text("learningSituationTitle", state.title);
  text("learningSituationReason", state.reason);
  text("learningSituationNext", state.next);
  text("learningFlowHealth", state.reason + " " + state.next);
  property("learningErrorNotice", "hidden", !state.error);
  text("learningErrorNotice", state.error ? "학습 오류: " + state.error : "");
  const last = ["champion", "candidate"]
    .map((role) => {
      const label = role === "champion" ? "Champion" : "Candidate",
        row = m[role + "_last_completed_round"];
      return (
        label +
        ": " +
        (row?.completed_utc
          ? timeOf(row.completed_utc) + " · " + whole(row.samples) + "건"
          : "완료 기록 미확인")
      );
    })
    .join(" / ");
  text("learningSituationLast", last);
}

function renderLearnerSummary(d) {
  const m = d.metrics || {};
  for (const role of ["champion", "candidate"]) {
    const round = m[role + "_last_completed_round"];
    const active = !!m[role + "_training"];
    const remaining = m[role + "_eligible_replay_count"];
    const state = learningSituation(d);
    const label =
      d.agent_process_running !== true
        ? d.agent_process_running === false
          ? "학습 프로세스 정지"
          : "학습 프로세스 미확인"
        : d.learning_enabled === false
          ? "학습 OFF"
          : d.learning_enabled !== true
            ? "학습 설정 미확인"
            : state.error
              ? "학습 오류 확인 필요"
              : active
                ? m.learning_wait_reason
                  ? "학습 회차 진행 · 연산 대기"
                  : "학습 중"
                : remaining == null
                  ? "잔여량 미확인"
                  : remaining > 0
                    ? m.learning_wait_reason
                      ? "처리 순서 대기"
                      : "다음 학습 준비"
                    : num(m.replay_pending_count ?? m.pending_experiences) > 0
                      ? "준비된 경험 처리 완료 · 결과 대기"
                      : "준비된 경험 처리 완료 · 새 경험 대기";
    text("learner" + role + "State", label);
    text(
      "learner" + role + "Work",
      "남은 경험 " +
        (remaining == null ? "미측정" : whole(remaining) + "건") +
        (active
          ? " · 현재 " + whole(m[role + "_samples_current"]) + "건 처리"
          : ""),
    );
    text(
      "learner" + role + "Last",
      round
        ? "최근 완료 " +
            timeOf(round.completed_utc) +
            " · " +
            whole(round.samples) +
            "건 · " +
            decimal(round.total_seconds, 1) +
            "초 · " +
            decimal(round.samples_per_total_second, 1) +
            "건/초"
        : "아직 완료된 학습 회차 없음",
    );
    text(
      "learner" + role + "Loss",
      round?.loss_mean == null ? "미측정" : decimal(round.loss_mean, 5),
    );
  }
}

function renderTrialSummary(d) {
  const v = d.validation_comparison || {};
  const bars = num(v.bars_current),
    required = num(v.bars_required || 390);
  property("trialProgressMeter", "max", required);
  property("trialProgressMeter", "value", Math.min(bars, required));
  text(
    "trialProgressLabel",
    whole(bars) +
      " / " +
      whole(required) +
      "개 시장 시점 · " +
      decimal(100 * Math.min(bars / required, 1), 1) +
      "%",
  );
  const scores = {};
  for (const role of ["champion", "candidate"]) {
    const books = v[role] || {};
    const rates = ["KRW", "USD"].map((c) => books[c]?.net_return_rate);
    const known =
      v.accounts_available &&
      rates.every((x) => x != null && Number.isFinite(Number(x)));
    if (known) {
      const trades = ["KRW", "USD"].reduce(
        (n, c) => n + num(books[c]?.trade_count),
        0,
      );
      const positions = ["KRW", "USD"].reduce(
        (n, c) => n + num(books[c]?.position_count),
        0,
      );
      const decisions = v.last_decisions?.[role] || [];
      const buys = decisions.filter((x) => x.action === "BUY").length;
      const holds = decisions.filter((x) => x.action === "HOLD").length;
      const sells = decisions.filter((x) => x.action === "SELL").length;
      text(
        role === "champion"
          ? "trialChampionActivity"
          : "trialCandidateActivity",
        "시험 체결 " +
          whole(trades) +
          "건 · 보유 " +
          whole(positions) +
          "종목" +
          (decisions.length
            ? "\n최근 입력 종목 판단: 매수 " +
              buys +
              " · 관망 " +
              holds +
              " · 매도 " +
              sells
            : ""),
      );
    } else
      text(
        role === "champion"
          ? "trialChampionActivity"
          : "trialCandidateActivity",
        "시험 계좌 미확인",
      );
    scores[role] = known ? rates.reduce((s, x) => s + Number(x), 0) / 2 : null;
    text(
      role === "champion" ? "trialChampionScoreNow" : "trialCandidateScoreNow",
      known ? (scores[role] * 100).toFixed(4) + "%" : "시험 계좌 미확인",
    );
  }
  const matching = [
    "same_starting_cash",
    "same_market_input",
    "same_market_timeline",
    "same_last_bar",
    "same_cost_rules",
    "same_action_rule",
  ].every((key) => v[key]);
  let leader;
  if (v.active && d.observe_enabled === false)
    leader = "판단 OFF · 승급전 일시정지";
  else if (v.active && !matching) leader = "대결 진행 · 동일 조건 확인 필요";
  else if (v.active && scores.champion != null && scores.candidate != null)
    leader =
      scores.candidate === scores.champion
        ? "현재 동률 · 진행 중"
        : (scores.candidate > scores.champion
            ? "현재 Candidate 앞섬"
            : "현재 Champion 앞섬") + " · 잠정 성과";
  else if (v.status === "promoted" && v.comparison_valid)
    leader = "판정 완료 · Candidate 승급";
  else if (v.status === "rejected" && v.comparison_valid)
    leader = "판정 완료 · Champion 유지";
  else
    leader =
      v.status === "discarded" || v.status === "rejected"
        ? "이전 시험 무효 · 새 대결 대기"
        : "새 고정 시험본 대결 대기";
  text("trialLeader", leader);
  text("trialNextAction", trialNextAction(d));
}

// Operator summaries read the same API snapshot as the detailed tables.
function pendingOutcomeState(d) {
  const m = d.metrics || {},
    seconds = Number(m.reward_credit?.duration_seconds);
  const labels = {
    next_quote: "새 시세",
    reward_horizon: "손익 확인 시간 경과",
    fill: "가상 주문 체결",
    missing_market_input: "시세 입력 복구",
    blocked: "기록 오류 해결",
  };
  const reasons = {};
  for (const role of ["champion", "candidate"])
    for (const [key, count] of Object.entries(
      m[role + "_pending_reward_status"]?.reasons || {},
    ))
      reasons[key] = (reasons[key] || 0) + num(count);
  const rows = Object.entries(reasons).filter(([, count]) => count > 0);
  const count = m.replay_pending_count ?? m.pending_experiences;
  const accounted = rows.reduce((sum, [, n]) => sum + n, 0);
  const detail = rows
    .map(
      ([key, value]) =>
        (labels[key] || "원인 미확인") + " · " + whole(value) + "건",
    )
    .join(" / ");
  const updating = count != null && accounted !== Number(count);
  return {
    window:
      Number.isFinite(seconds) && seconds > 0
        ? "현재 설정 " + whole(seconds / 60) + "분"
        : "기간 미확인",
    reason: rows.length
      ? detail +
        (updating
          ? " / 사유 집계 갱신 중" +
            (Number(count) > accounted
              ? " · " + whole(Number(count) - accounted) + "건"
              : "")
          : "")
      : count == null
        ? "상태 미확인"
        : Number(count) === 0
          ? "대기 없음"
          : "대기 이유 미확인",
  };
}
function trialNextAction(d) {
  const v = d.validation_comparison || {};
  if (d.status_unavailable)
    return "서버 연결 끊김 · 승급전 상태를 확인할 수 없습니다.";
  if (!d.agent_process_running)
    return "모델 프로세스가 정지했습니다. 시작 후 시험을 진행합니다.";
  if (v.active && d.observe_enabled === false)
    return "모델 판단 OFF · 판단을 켜면 시험도 다시 진행됩니다.";
  if (v.active) {
    const required = Number(v.bars_required),
      current = Number(v.bars_current);
    const remaining =
      Number.isFinite(required) && Number.isFinite(current)
        ? Math.max(0, required - current)
        : null;
    return (
      (remaining == null
        ? "시험 진행량 확인 중"
        : "평가할 시장 시점 " + whole(remaining) + "개 남음") +
      " · 새 시세를 받으며 평가 → 일일 판정 " +
      (d.daily_cycle?.next_reset_utc
        ? timeOf(d.daily_cycle.next_reset_utc)
        : "시각 미확인") +
      " · 장기 가상계좌는 유지"
    );
  }
  if (v.comparison_valid && ["promoted", "rejected"].includes(v.status))
    return "판정 완료 · 다음 시험은 새 고정 모델과 별도 시험계좌로 시작합니다. 장기 가상계좌는 유지됩니다.";
  return "새 시험본 준비 대기 · 같은 시장·시작 자금·비용으로 두 모델을 비교합니다.";
}
function renderWorkflowStatus(d) {
  const m = d.metrics || {},
    h = d.agent_health || {},
    c = h.candidate || {};
  if (d.status_unavailable) {
    for (const id of [
      "workflowFeed",
      "workflowInference",
      "workflowPaper",
      "workflowTrial",
    ])
      text(id, "연결 끊김 · 미확인");
    return;
  }
  text(
    "workflowFeed",
    d.feed_running
      ? "수집 중 · " +
          whole((d.feed_metrics?.fresh_symbols_5m || []).length) +
          "종목 수신"
      : "정지",
  );
  text(
    "workflowInference",
    !d.agent_process_running
      ? "프로세스 정지"
      : d.observe_enabled === false
        ? "OFF"
        : h.status === "error" || c.status === "error"
          ? "오류 · 확인 필요"
          : h.lag_seconds == null || c.lag_seconds == null
            ? "지연 미측정"
            : "ON · 시세 처리 지연 " +
              whole(Math.max(Number(h.lag_seconds), Number(c.lag_seconds))) +
              "초",
  );
  text(
    "workflowPaper",
    d.paper_enabled == null
      ? "상태 미확인"
      : d.paper_enabled
        ? "ON · 가상계좌 체결"
        : "OFF",
  );
  const v = d.validation_comparison || {};
  text(
    "workflowTrial",
    v.active
      ? !d.agent_process_running
        ? "프로세스 정지 · 중지"
        : d.observe_enabled === false
          ? "판단 OFF · 중지"
          : whole(v.bars_current) + " / " + whole(v.bars_required) + "시점"
      : v.status === "promoted" && v.comparison_valid
        ? "Candidate 승급 완료"
        : v.status === "rejected" && v.comparison_valid
          ? "Champion 유지"
          : "시험 대기",
  );
}
