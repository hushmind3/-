"use strict";

// Long-lived paper accounts: summary, positions, actions and fills.

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

function renderCandidateLiveAccount(d, summaryOnly = false) {
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
    positions = summaryOnly
      ? []
      : ["KRW", "USD"].flatMap((currency) =>
          (books[currency]?.positions || []).map((p) => ({ ...p, currency })),
        ),
    decisions = summaryOnly
      ? ""
      : (v.decisions || [])
          .map((x) => instrumentLabel(x.symbol, d) + " " + x.action)
          .join(" · ") || "최근 판단 없음",
    status = !modelIsRunning(d, "candidate")
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
  text("candidateLiveKrwNet", money(krw) + " KRW");
  text("candidateLiveUsdNet", money(usd) + " USD");
  text("candidateLiveKrwSplit", split(krw) + " KRW");
  text("candidateLiveUsdSplit", split(usd) + " USD");
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
  if (summaryOnly) return;
  text("candidateLiveSeed", currencyPair(books, "initial_cash"));
  text("candidateLiveCash", currencyPair(books, "cash"));
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
  const l = d.learning || {};
  if (!d.account_observability) {
    text(
      "paperTradeCount",
      whole(l.paper_trade_count) +
        "건 (KRW " +
        whole(l.paper_trade_counts_by_currency?.KRW) +
        " · USD " +
        whole(l.paper_trade_counts_by_currency?.USD) +
        ")",
    );
  }

  if (d.account_observability) {
    renderCandidateLiveAccount(d, true);
    renderAccountDiagnostics(d);
    renderLiveAccountComparison(d);
    return;
  }
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
  renderCandidateLiveAccount(d);
  renderLiveAccountComparison(d);
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
    const lifecycle=d.model_runtime?.[role];
    const active = modelIsRunning(d, role),
      enabled = Boolean(d.paper_enabled);
    const badge = isChampion ? "championLiveBadge" : "candidateLiveBadge";
    const error = lifecycle?.error || active && (account.observer_error || health?.status === "error");
    text(
      badge,
      lifecycle
        ? modelStateLabel(lifecycle)
        : error
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
    if (lifecycle) {
      text(role + "ModelResidency", (lifecycle.loaded ? "모델 적재됨" : "모델 메모리 해제됨") + " · RAM " + (lifecycle.memory_scope === "worker" ? "프로세스 사용 " : "가중치 ") + decimal((lifecycle.ram_weight_bytes || 0)/1024**3,2) + " GB · GPU 가중치 " + decimal((lifecycle.gpu_weight_bytes || 0)/1024**3,2) + " GB" + (lifecycle.loaded ? " · 계산 장치 " + (lifecycle.compute_device || lifecycle.device || "—") : ""));
      text(role + "ModelDecision", (modelUsesHistoricalData(d,role) ? "과거 데이터 " + marketDataTime(account.last_full_decision_timestamp) : "최근 판단 " + timeOf(account.last_full_decision_timestamp)) + " · 계산 " + (account.last_inference_seconds == null ? "—" : decimal(account.last_inference_seconds,2) + "초"));

    }
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
    for (const currency of ["KRW", "USD"]) {
      const b = books[currency],
        prefix = role + currency;
      text(prefix + "Return", accountPercent(b?.net_return_rate));
      property(
        prefix + "Return",
        "className",
        b?.net_pnl == null
          ? ""
          : num(b.net_pnl) < 0
            ? "account-negative"
            : "account-positive",
      );
      text(prefix + "Pnl", accountMoney(b?.net_pnl, currency));
      property(
        prefix + "Pnl",
        "className",
        "net-value " +
          (b?.net_pnl == null
            ? ""
            : num(b.net_pnl) < 0
              ? "account-negative"
              : "account-positive"),
      );
      text(
        prefix + "Equity",
        "현재 순자산 " + accountMoney(b?.equity, currency),
      );
      text(
        prefix + "Exposure",
        "현금 " +
          accountPercent(b?.cash_ratio) +
          " · 최대 비중 " +
          accountPercent(b?.largest_position_weight),
      );
      text(
        prefix + "Trades",
        b?.trade_count == null
          ? "체결 미확인"
          : !num(b.trade_count)
            ? "아직 체결 없음"
            : "누적 " + whole(b.trade_count) + "건 체결",
      );
    }
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
    if (detailsOpen(isChampion ? "positions" : "candidateLiveHoldings")) {
      const positions = ["KRW", "USD"].flatMap((c) =>
        (books[c]?.positions || []).map((p) => [
          instrumentLabel(p.symbol, d),
          c,
          p.quantity_unit ? decimal(p.quantity,3) + " " + p.quantity_unit : whole(p.quantity) + "주",
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
    }
    if (detailsOpen(role + "CostDetails")) {
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
    }
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
    if (detailsOpen(isChampion ? "championLiveFills" : "candidateLiveFills")) {
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
  }
}

function renderPaperResults(d) {
  const m = d.metrics || {},
    pf = d.paper_financials || {},
    format = (value, currency) =>
      value == null ? "—" : Number(value).toFixed(2) + " " + currency,
    krw = pf.KRW || {},
    usd = pf.USD || {};
  text("krwNetPnl", format(krw.net_pnl, "KRW"));
  text("usdNetPnl", format(usd.net_pnl, "USD"));
  text(
    "krwPnlSplit",
    format(krw.realized_pnl, "KRW") + " / " + format(krw.unrealized_pnl, "KRW"),
  );
  text(
    "usdPnlSplit",
    format(usd.realized_pnl, "USD") + " / " + format(usd.unrealized_pnl, "USD"),
  );
  const cost = (b) =>
    Number(b.fees || 0) +
    Number(b.sell_tax || 0) +
    Number(b.spread || 0) +
    Number(b.slippage || 0);
  if (!d.account_observability)
    text(
      "paperCosts",
      format(cost(krw), "KRW") + " | " + format(cost(usd), "USD"),
    );
}

function mInferenceProfile(d, role) {
  return d.metrics?.[role + "_live_inference_profile"] || {};
}
