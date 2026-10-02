"use strict";

// Frozen promotion trial: its accounts, progress, fairness and daily decisions.

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

// Same metric definitions on every view; no timer-driven layout rebuilding.

function renderPromotionHistory(d) {
  const m = d.metrics || {},
    v = d.validation_comparison || {},
    cycle = d.daily_cycle || {};
  const score = (role) =>
      Object.values(v[role] || {}).reduce(
        (total, b) => total + num(b.net_return_rate),
        0,
      ),
    cs = score("candidate"),
    bs = score("champion"),
    required = num(v.bars_required) || 390,
    bars = num(v.bars_current);
  const health =
    v.same_market_input &&
    v.same_market_timeline &&
    v.same_starting_cash &&
    v.same_cost_rules &&
    v.same_action_rule;
  text(
    "promotionDailyConditions",
    "하루 승급전 · Champion 고정 v" +
      whole(v.champion_snapshot_version) +
      " / Candidate 고정 v" +
      whole(v.snapshot_version) +
      " · 시작 " +
      timeOf(v.started_utc || v.start_after) +
      " · 관측 " +
      whole(bars) +
      " / 최소 " +
      whole(required) +
      "개 시장 분. 통화별 시드 대비 수익률 합: Champion " +
      accountPercent(bs) +
      " / Candidate " +
      accountPercent(cs) +
      ". 조건: Candidate 이익 " +
      (cs > 0 ? "통과" : "미충족") +
      " · Champion 초과 " +
      (cs > bs ? "통과" : "미충족") +
      " · 최소 관측 " +
      (bars >= required ? "통과" : "미충족") +
      " · 동일 입력/비용 " +
      (health ? "확인" : "대기/불일치") +
      ". 중간 수익만으로 즉시 승급하지 않고 오전 " +
      whole(cycle.hour_kst ?? 7) +
      "시에 판정합니다.",
  );
  const reason = (value) =>
    ({
      "candidate paper-account net return was not positive":
        "Candidate 비용 차감 이익 없음",
      "candidate paper-account net return did not beat champion":
        "Champion 초과 성과 없음",
      "sequential paper-account net return improved":
        "이익을 내고 Champion을 이겨 승급",
      "sequential paper validation window is incomplete": "최소 시장 관측 부족",
      "restart invalidated the in-memory validation snapshot":
        "재시작으로 RAM 시험본 무효",
    })[value] ||
    value ||
    "상세 사유 미기록";
  const history = (m.candidate_gate_history || []).slice().reverse();
  if ($("promotionDecisionHistory").closest("details")?.open)
    html("promotionDecisionHistory", () =>
      history.length
        ? accountTable(
            [
              "판정 시각",
              "Champion / Candidate 버전",
              "시장 관측",
              "두 점수 · Champion / Candidate",
              "결과",
              "사유",
            ],
            history.map((h) => [
              timeOf(h.time_utc),
              h.champion_version == null
                ? "과거 미기록"
                : "v" +
                  whole(h.champion_version) +
                  " / v" +
                  whole(h.candidate_version),
              h.bars == null
                ? "과거 미기록"
                : whole(h.bars) + " / " + whole(h.required_bars),
              accountPercent(h.champion_score) +
                " / " +
                accountPercent(h.candidate_score),
              h.applied ? "승급" : "유지",
              reason(h.reason),
            ]),
          )
        : "새 하루 승급전 결과 대기",
    );
}
