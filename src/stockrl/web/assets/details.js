"use strict";

// Replay, learning, model metrics, promotion trials and expandable diagnostics.

function renderLearningTotals(d) {
  const m = d.metrics || {};
  text("replayCount", whole(m.replay_count));

  text("maturedCount", whole(m.matured));
  text("updateCount", whole(m.updates));
  text("lastUpdate", "마지막 " + timeOf(m.last_update_utc));
  text(
    "gateCount",
    "새 모델 적용 " +
      whole(m.promotions) +
      "회 · 적용 안 함 " +
      whole(m.rejections) +
      "회",
  );
  const training = !!m.candidate_training;

  property(
    "paperReward",
    "className",
    num(m.paper_net_reward) < 0 ? "value-negative" : "value-positive",
  );
  text("feeTotal", decimal(m.fee_total, 4));
  text("slippageTotal", decimal(m.slippage_total, 4));

  text("lastUpdateFull", timeOf(m.last_update_utc));
  text("lastPromotion", timeOf(m.last_promotion_utc));
  text("lastRejection", timeOf(m.last_rejection_utc));
  const promoted = Date.parse(m.last_promotion_utc || "") || 0,
    rejected = Date.parse(m.last_rejection_utc || "") || 0,
    hasGate = promoted || rejected,
    applied = promoted > rejected,
    cs = m.last_candidate_validation_score,
    bs = m.last_champion_validation_score;
  const verdict = !hasGate
    ? "아직 비교 전"
    : applied
      ? "새 모델 적용 · 현재 판단에 사용 중"
      : "새 모델 적용 안 함 · 이전 모델 계속 사용";
  text("lastGateResult", verdict);

  text(
    "inferenceTime",
    m.inference_seconds_p50 == null
      ? "—"
      : (num(m.inference_seconds_p50) * 1000).toFixed(0) + " ms",
  );
  const comparison =
    cs == null || bs == null
      ? "검증 점수 기록 없음"
      : "새 모델 " +
        decimal(cs, 5) +
        " / 기존 모델 " +
        decimal(bs, 5) +
        (Number(cs) > Number(bs)
          ? " · 새 모델 점수가 더 높음"
          : " · 기존 모델 점수가 같거나 더 높음");
  html(
    "gateStory",
    () =>
      "<strong>" +
      (training
        ? "지금 새 모델 학습 중 · 기존 모델은 계속 판단합니다."
        : verdict) +
      "</strong>" +
      esc(comparison) +
      "<br>적용 " +
      whole(m.promotions) +
      "회, 적용 안 함 " +
      whole(m.rejections) +
      "회는 누적 자동 검증 횟수입니다. 직접 누르는 승인 버튼이 아닙니다.",
  );
  const history = Array.isArray(m.candidate_gate_history)
    ? m.candidate_gate_history.slice().reverse()
    : [];
  if ($("gateHistory").closest("details")?.open)
    html("gateHistory", () =>
      history.length
        ? history
            .map(
              (h) =>
                "<div><strong>" +
                esc(timeOf(h.time_utc)) +
                " · " +
                (h.applied ? "새 모델 적용" : "기존 모델 유지") +
                "</strong><br>" +
                esc(h.reason || "같은 구간 순보상 비교") +
                " · 새 모델 " +
                decimal(h.candidate_score, 5) +
                " / 기존 모델 " +
                decimal(h.champion_score, 5) +
                "</div>",
            )
            .join("")
        : "<div>마지막 비교: " +
          esc(verdict) +
          " · " +
          esc(comparison) +
          "</div><div>이전 개별 비교의 상세 기록은 저장되어 있지 않습니다. 위 누적 횟수는 실제 저장된 집계입니다.</div>",
    );
}

function renderPaperResults(d) {
  const m = d.metrics || {},
    l = d.learning || {},
    pf = d.paper_financials || {},
    format = (value, currency) =>
      value == null ? "—" : Number(value).toFixed(2) + " " + currency,
    krw = pf.KRW || {},
    usd = pf.USD || {};
  text(
    "paperReward",
    m.paper_net_reward == null
      ? "—"
      : (Number(m.paper_net_reward) * 100).toFixed(3) + "% (비용 차감)",
  );
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
  text(
    "paperCosts",
    format(cost(krw), "KRW") + " | " + format(cost(usd), "USD"),
  );

  const status = !l.promotion_gate_ready
    ? "자동 승급 차단: " + (l.promotion_blocked_reason || "검증 조건 미완료")
    : l.candidate_stage === "sequential_paper_validation"
      ? "새 paper 계좌 비교 중"
      : "순차 paper 계좌 비교 준비 완료";

  if (!l.promotion_gate_ready && !m.last_promotion_utc && !m.last_rejection_utc)
    text("lastGateResult", "비교 전 | 승급 차단");
  if (!l.promotion_gate_ready && !m.last_promotion_utc && !m.last_rejection_utc)
    text("lastGateResult", "비교 전 | 승급 차단");
}

function renderModelComparison(d) {
  const m = d.metrics || {},
    v = d.validation_comparison || {},
    required = Number(v.bars_required) || 390;
  const seconds = (x) => (x == null ? "미측정" : Number(x).toFixed(2) + "초"),
    bytes = (x) =>
      x == null ? "미측정" : (Number(x) / 1073741824).toFixed(2) + " GiB",
    score = (x) => (x == null ? "미측정" : (Number(x) * 100).toFixed(4) + "%");
  for (const role of ["champion", "candidate"]) {
    const title = role === "champion" ? "Champion" : "Candidate",
      r = m[role + "_last_completed_round"] || {},
      candidate = role === "candidate";
    const params = candidate ? m.candidate_total_parameter_count : m.parameters,
      trainable = m[role + "_trainable_parameter_count"];
    text(
      "compare" + title + "Params",
      params == null ? "미측정" : whole(params) + "개",
    );
    text(
      "compare" + title + "Trainable",
      trainable == null ? "미측정" : whole(trainable) + "개",
    );
    const enabled = m[role + "_learning_enabled"],
      training = m[role + "_training"];
    text(
      "compare" + title + "LearningState",
      !d.agent_process_running
        ? "정지 · 마지막 기록"
        : d.learning_enabled === false
          ? "학습 OFF · replay 보존"
          : !enabled
            ? "학습 비활성"
            : training
              ? "학습 중 · optimizer " +
                whole(m[role + "_optimizer_steps_current"]) +
                "회"
              : "학습 활성 · 회차 사이 대기",
    );
    const changes =
        m[candidate ? "weight_delta_l1" : "champion_weight_delta_l1"],
      delta =
        Array.isArray(changes) && changes.length
          ? changes[changes.length - 1]
          : null,
      version = r.model_version;
    text(
      candidate ? "compareWeightDelta" : "compareChampionWeightDelta",
      (version == null
        ? "학습 버전 미기록"
        : "최근 학습 완료 v" + whole(version)) +
        " · L1 " +
        (delta == null ? "미측정" : Number(delta).toExponential(3)),
    );
    text(
      "compare" + title + "Compute",
      whole(m[role + "_live_inference_count"]) +
        "회 · 경과 시간 누계 " +
        seconds(m[role + "_live_inference_seconds_total"]),
    );
    text(
      "compare" + title + "Learning",
      r.completed_utc
        ? whole(r.unique_samples) +
            "개 경험 · optimizer " +
            whole(r.optimizer_steps) +
            "회 · 계산 " +
            seconds(r.compute_seconds) +
            " / 총 " +
            seconds(r.total_seconds) +
            " · " +
            timeOf(r.completed_utc)
        : "완료 회차 미측정",
    );
    text(
      "compare" + title + "Gpu",
      r.completed_utc
        ? "allocated " +
            bytes(r.peak_allocated_bytes) +
            " · reserved " +
            bytes(r.peak_reserved_bytes)
        : "완료 회차 미측정",
    );
  }
  const history = m.candidate_gate_history || [],
    finished = [...history]
      .reverse()
      .find(
        (h) =>
          h.candidate_version != null &&
          h.champion_version != null &&
          Number(h.required_bars) >= required &&
          Number(h.bars) >= Number(h.required_bars) &&
          h.candidate_score != null &&
          h.champion_score != null &&
          Number.isFinite(Number(h.candidate_score)) &&
          Number.isFinite(Number(h.champion_score)),
      );
  text(
    "compareChampionScore",
    finished ? score(finished.champion_score) : "현재 기준 완료 기록 없음",
  );
  text(
    "compareCandidateScore",
    finished ? score(finished.candidate_score) : "현재 기준 완료 기록 없음",
  );
  text(
    "compareWinner",
    finished
      ? timeOf(finished.time_utc) +
          " · 고정 Champion v" +
          whole(finished.champion_version) +
          " / Candidate v" +
          whole(finished.candidate_version) +
          " · " +
          (finished.applied ? "Candidate 승급" : "Champion 유지") +
          " · " +
          (Number(finished.candidate_score) > Number(finished.champion_score)
            ? "Candidate 점수 우세"
            : Number(finished.candidate_score) < Number(finished.champion_score)
              ? "Champion 점수 우세"
              : "동점")
      : "현재 최소 관측 기준을 마친 승급전 기록이 없습니다. 과거 운영 기록은 상세 이력에서 확인합니다.",
  );
  const currentScore = (role) => {
    const books = Object.values(v[role] || {});
    return books.length &&
      books.every((b) => Number.isFinite(b.net_return_rate))
      ? books.reduce((sum, b) => sum + b.net_return_rate, 0)
      : null;
  };
  const progress =
    whole(v.bars_current) +
    " / " +
    whole(required) +
    "개 시장 분 · " +
    (d.observe_enabled === false && v.active
      ? "승급전 일시정지 · 판단 OFF"
      : v.active
        ? "진행 중 · 미확정"
        : "시험 비활성 · 마지막 상태");
  text(
    "compareProgress",
    progress +
      " · 고정 Champion v" +
      whole(v.champion_snapshot_version) +
      " " +
      score(currentScore("champion")) +
      " / Candidate v" +
      whole(v.snapshot_version) +
      " " +
      score(currentScore("candidate")) +
      " · 계좌 시각 " +
      timeOf(v.last_timestamp),
  );
  text(
    "modelComparisonExplanation",
    "두 모델 모두 판단·가상매매·학습합니다. 누적 횟수는 과거 실행 기록까지 포함해 같은 시작 시각의 대결 횟수가 아닙니다. 누적 판단 시간의 측정 범위도 다릅니다: Champion은 입력 준비 후 계산, Candidate는 대기·이동까지 포함합니다. 최근 학습은 양쪽 동일한 항목으로 표시합니다. VRAM은 해당 회차 중 전체 프로세스 최고치입니다. L1은 가중치 변화량이며 수익률이나 성장 점수가 아닙니다. 완료 점수는 고정본 버전과 관측 길이가 확인된 최근 승급전 기록이고, 진행 중 점수는 현재 시험 계좌에서 계산합니다.",
  );
}

function renderLearningMeasurements(d) {
  renderModelComparison(d);
  const m = d.metrics || {},
    l = d.learning || {},
    gb = (v) => (v == null ? "—" : (Number(v) / 1073741824).toFixed(2) + " GB"),
    pct = (v) => (v == null ? "—" : (Number(v) * 100).toFixed(4) + "%"),
    missing = "아직 측정 전";
  text(
    "paperTradeCount",
    whole(l.paper_trade_count) +
      "건 (KRW " +
      whole(l.paper_trade_counts_by_currency?.KRW) +
      " · USD " +
      whole(l.paper_trade_counts_by_currency?.USD) +
      ")",
  );
  text(
    "currentValidationProgress",
    whole(l.validation_bars_current) +
      " / " +
      whole(l.validation_bars_required) +
      " bar",
  );

  text(
    "candidateTrainingVram",
    l.candidate_peak_allocated_bytes == null
      ? missing
      : "최고 " +
          gb(l.candidate_peak_allocated_bytes) +
          " · 시작 " +
          (l.candidate_baseline_allocated_bytes == null
            ? "미측정"
            : gb(l.candidate_baseline_allocated_bytes)) +
          " · 예약 " +
          gb(l.candidate_peak_reserved_bytes),
  );
  const currentGpuAllocated = l.gpu_allocated_bytes,
    currentGpuReserved = l.gpu_reserved_bytes;
  text(
    "modelGpuMemoryCurrent",
    currentGpuAllocated == null
      ? missing
      : gb(currentGpuAllocated) +
          " allocated / " +
          (currentGpuReserved == null ? "n/a" : gb(currentGpuReserved)) +
          " reserved",
  );
  const weightDeltas = m.weight_delta_l1;
  const lastDelta =
    Array.isArray(weightDeltas) && weightDeltas.length
      ? Number(weightDeltas[weightDeltas.length - 1])
      : null;
  text(
    "candidateWeightDelta",
    lastDelta == null
      ? missing
      : lastDelta === 0
        ? "변화 없음 (0)"
        : "변경됨 (L1) " + lastDelta.toExponential(3),
  );
  text("championWeightsBytes", gb(m.champion_model_parameter_bytes));
  text(
    "championInferenceUsage",
    whole(m.champion_live_inference_count) +
      "회 · " +
      num(m.champion_live_inference_seconds_total).toFixed(1) +
      "초 · p50/p95 " +
      (num(m.inference_seconds_p50) * 1000).toFixed(0) +
      "/" +
      (num(m.inference_seconds_p95) * 1000).toFixed(0) +
      "ms",
  );
  text(
    "championValidationUsage",
    whole(m.champion_validation_inference_count) +
      "회 · " +
      num(m.champion_validation_inference_seconds_total).toFixed(1) +
      "초",
  );
  text(
    "candidateValidationUsage",
    whole(m.candidate_validation_inference_count) +
      "회 · " +
      num(m.candidate_validation_inference_seconds_total).toFixed(1) +
      "초",
  );
  text(
    "candidateScore",
    l.candidate_validation_score == null
      ? "—"
      : pct(l.candidate_validation_score),
  );
  text(
    "championScore",
    l.champion_validation_score == null
      ? "—"
      : pct(l.champion_validation_score),
  );
  const phase =
    d.learning_enabled === false
      ? "학습 OFF · replay 보존"
      : l.candidate_stage === "sequential_paper_validation"
        ? "검증 비교 진행 중"
        : l.candidate_stage === "training"
          ? "candidate 학습 중"
          : l.candidate_learning_enabled
            ? "candidate 학습 대기"
            : "candidate 자동 학습 꺼짐";
  const scores =
    l.candidate_validation_score == null || l.champion_validation_score == null
      ? "완료된 비교 점수 없음"
      : "candidate " +
        pct(l.candidate_validation_score) +
        " / champion " +
        pct(l.champion_validation_score);
  const gate = l.promotion_gate_ready
    ? "승급 비교 조건 충족"
    : "승급 대기: " +
      (l.promotion_blocked_reason ||
        l.candidate_skip_reason ||
        "검증 조건 미충족");

  if (l.candidate_skip_reason)
    text(
      "learningThreshold",
      "검증 진행 " +
        whole(l.validation_bars_current) +
        " / " +
        whole(l.validation_bars_required) +
        " bar · 학습 표본 " +
        whole(l.replay_current) +
        " / " +
        whole(l.candidate_every) +
        " · 대기 이유: " +
        l.candidate_skip_reason,
    );
}

function renderLearningMetrics(d) {
  const m = d.metrics || {},
    h = d.agent_health || {},
    candidateHealth = h.candidate || {},
    lag = (value) =>
      value == null
        ? !d.agent_process_running
          ? "정지"
          : "미측정"
        : num(value).toFixed(0) + "초",
    training = !!m.candidate_training,
    trial = !!m.candidate_validation_active,
    trialPaused = trial && d.observe_enabled === false;
  const progress = trialPaused
    ? "승급전 일시정지 · 시장 관찰 OFF"
    : training
      ? "Candidate 학습 중 · optimizer " +
        whole(m.candidate_optimizer_steps_current) +
        " / " +
        whole(m.candidate_optimizer_steps_target) +
        "회 · " +
        whole(m.candidate_samples_current) +
        "개 replay 입력 처리"
      : m.candidate_skip_reason || "새 경험 대기";

  text(
    "candidateOptimizerCount",
    whole(
      training
        ? m.candidate_optimizer_steps_current
        : m.last_candidate_optimizer_steps,
    ) +
      " / " +
      whole(m.candidate_optimizer_steps_target) +
      "회",
  );
  text(
    "candidateTrainingSamples",
    whole(
      training ? m.candidate_samples_current : m.last_candidate_samples_trained,
    ) +
      "개 · 목표 " +
      whole(m.candidate_samples_target) +
      "개",
  );
  const round = m.candidate_last_completed_round;
  text(
    "candidateTrainingTime",
    round
      ? "최근 완료: 총 " +
          num(round.total_seconds).toFixed(1) +
          "초 · 계산 " +
          num(round.compute_seconds).toFixed(1) +
          "초 · 단계당 " +
          num(round.step_compute_seconds).toFixed(1) +
          "초"
      : m.candidate_training
        ? "학습 중 · 이번 회차 완료 후 시간 측정"
        : "아직 측정 전",
  );
  const coverage = m.multiscale_coverage || {};
  text(
    "appliedSettings",
    "최근 입력 기록 확보: " +
      Object.entries(coverage)
        .map(
          ([k, v]) =>
            (({
              "1m": "1분",
              "3m": "3분",
              "5m": "5분",
              "15m": "15분",
              "60m": "60분",
              "1d": "일",
              "1w": "주",
              "1mo": "월",
            })[k] || k) +
            " " +
            (Number(v) * 100).toFixed(0) +
            "%",
        )
        .join(" · ") +
      " · 매수/매도 호가 제공 " +
      whole(m.quoted_bid_ask_symbols) +
      "종목 · 마지막 전체 추론 기준 " +
      (m.multiscale_input_status_utc
        ? timeOf(m.multiscale_input_status_utc)
        : "미측정"),
  );
  const b = d.backtest || {};
  text(
    "portfolioBacktestSummary",
    b.evaluation_mode
      ? "최근 계좌 백테스트: " +
          whole(b.timestamps) +
          "구간 · 체결 " +
          whole(b.trade_count) +
          "건 · 순자산 수익률 " +
          (num(b.net_return) * 100).toFixed(4) +
          "% · " +
          num(b.elapsed_seconds).toFixed(1) +
          "초. 과거 진단이며 승급 판정에는 사용하지 않습니다."
      : "계좌 백테스트 기록 없음",
  );
  const error =
    m.agent_last_input_error ||
    m.last_candidate_error ||
    m.candidate_validation_error ||
    m.candidate_live_error;
  if (error) {
    text(
      "learningFlowHealth",
      textOf("learningFlowHealth") + " | 확인할 오류: " + error,
    );
  }
  text(
    "gateStory",
    trialPaused
      ? "승급전 일시정지 · 시장 관찰 OFF"
      : trial
        ? "고정 시험본의 새로운 미래 구간 비교 " +
          whole(m.candidate_validation_bars) +
          " / 390 · 동일 자금·비용의 순자산 성과로 판정"
        : m.candidate_validation_error
          ? "최근 시험 무효: " + m.candidate_validation_error
          : "새 미래 구간 시험 준비 · 마지막 완료 점수는 위 비교표 참조",
  );
}

function renderDailyLearning(d) {
  const m = d.metrics || {},
    rows = m.daily_learning || [],
    host = $("dailyLearningRows");
  const objective = m.shared_objective;
  if (objective) {
    const goals = ["champion", "candidate"].flatMap((role) => {
      const goal = m[role + "_goal"] || {};
      return Object.entries(goal.books || {}).map(([currency, b]) => [
        role === "champion" ? "Champion" : "Candidate",
        currency,
        num(b.multiple).toFixed(4) + "배",
        accountMoney(b.target_equity, currency),
        accountPercent(
          b.target_asset_ratio ??
            num(b.multiple) / num(goal.target_multiple || 10),
        ),
        accountPercent(b.net_return_rate ?? num(b.multiple) - 1),
        b.win ? "WIN · " + timeOf(b.win.timestamp) : "진행 중",
      ]);
    });
    const goalTraining = ["champion", "candidate"]
      .map((role) => {
        const round = m[role + "_last_completed_round"],
          label = role === "champion" ? "Champion" : "Candidate";
        return (
          label +
          ": " +
          (round?.goal_conditioned_samples == null
            ? "새 목표 입력의 완료 학습 회차는 아직 미확인"
            : "최근 완료 회차 " +
              whole(round.goal_conditioned_samples) +
              " / " +
              whole(round.samples) +
              "개 목표 입력 학습 · 성공 보너스 경험 " +
              whole(round.goal_bonus_samples) +
              "개")
        );
      })
      .join(" | ");
    html(
      "sharedGoalSummary",
      () =>
        "<strong>공통 임무: 비용 차감 순자산 " +
        esc(num(objective.target_multiple).toFixed(0)) +
        "배 · 두 모델 모두 순손익 + 목표 달성 경험으로 학습</strong><br>장기 계좌는 매일 유지합니다. 각 통화의 목표 달성은 episode당 한 번 기록하고 " +
        esc(whole(objective.win_bonus_points)) +
        "점의 학습용 성공 보상을 연결합니다. 승부 점수는 실제 계좌 성과이며 성공 보너스는 포함하지 않습니다." +
        "<br>목표 자산 대비 달성률 = 현재 순자산 ÷ 목표 순자산입니다. 학습률과 다릅니다. 10배 목표에서는 시작 자금이 10%이며 손실은 원금 대비 수익률에 표시합니다." +
        (goals.length
          ? accountTable(
              [
                "모델",
                "통화",
                "현재 자산 배율",
                "목표 순자산",
                "목표 자산 대비 달성률",
                "원금 대비 수익률",
                "목표 상태",
              ],
              goals,
            )
          : "<br>새 목표 방식 관측 대기") +
        "<br>" +
        esc(goalTraining),
    );
  }
  const credit = m.reward_credit;
  if (credit) {
    const scores = ["champion", "candidate"]
      .map((role) => {
        const score = m[role + "_reward_score"],
          p = score?.points || {};
        return (
          (role === "champion" ? "Champion" : "Candidate") +
          " 누적점수: " +
          (score
            ? "KRW " +
              num(p.KRW).toFixed(3) +
              "점 · USD " +
              num(p.USD).toFixed(3) +
              "점"
            : "새 방식 관측 대기")
        );
      })
      .join(" | ");
    text(
      "rewardCreditDescription",
      "비용 차감 계좌 수익률 +1% = +1점, −1% = −1점. " +
        scores +
        ". 새 보상은 직전 점수의 변화분입니다. 새 경험은 " +
        num(credit.duration_seconds / 60).toFixed(0) +
        "분 동안 관측한 누적 결과와 다음 상태의 예상 가치를 연결합니다. 입력봉 개수나 강제 보유시간 제한이 아닙니다. 초기화로 이전 계좌의 연결을 종료하며, 기존 짧은 보상 경험도 학습 후 삭제합니다.",
    );
  }
  const todayKey = new Intl.DateTimeFormat("sv-SE", {
      timeZone: "Asia/Seoul",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(new Date()),
    today = rows.find((row) => row.day === todayKey) || {};
  text("todayLearningGenerated", whole(today.enqueued));
  text("todayLearningCompleted", whole(today.completed));
  text("todayLearningRemaining", whole(today.remaining));
  text("allLearningRemaining", whole(m.replay_eligible_backlog));
  text(
    "blockedLearningCount",
    whole(
      Number(m.replay_quarantined_count || 0) +
        Number(m.replay_unsupported_count || 0),
    ),
  );
  const rate =
    Number(today.enqueued) > 0
      ? (100 * Number(today.completed || 0)) / Number(today.enqueued)
      : null;
  text(
    "learningCompletionSummary",
    rate == null
      ? "오늘 아직 보상 평가가 끝나 새로 저장된 경험이 없습니다."
      : "오늘 생성 " +
          whole(today.enqueued) +
          "건 → 학습·저장 완료 " +
          whole(today.completed) +
          "건 (" +
          rate.toFixed(2) +
          "%) · 학습 가능 잔여 " +
          whole(today.remaining) +
          "건 · 보류 " +
          whole(today.blocked) +
          "건. 완료 경험은 DB에서 삭제되어 현재 파일 건수와 누계가 다릅니다.",
  );
  text(
    "pendingLearningCount",
    whole(m.replay_pending_count ?? m.pending_experiences) + "건",
  );
  const blockedReasonNames = {
    "saved portfolio input shape is incompatible":
      "저장된 입력 형태 불일치 (세부 원인 기록 없음)",
    "unsupported reward schema": "지원하지 않는 과거 보상 형식",
  };
  text(
    "blockedLearningReason",
    Object.entries(m.replay_blocked_reasons || {})
      .map(
        ([reason, count]) =>
          (reason.startsWith("saved portfolio input shape is incompatible:")
            ? "저장 입력 형태 불일치 · " +
              reason.split(":").slice(1).join(":").trim()
            : blockedReasonNames[reason] || reason) +
          " " +
          whole(count) +
          "건",
      )
      .join(" · ") || "새 원인 계측 대기",
  );
  text("completedReplayRetained", whole(m.replay_completed_retained));
  text(
    "learningObservationTime",
    timeOf(m.last_market_timestamp) +
      " · " +
      (d.agent_health?.lag_seconds == null
        ? "지연 미확인"
        : num(d.agent_health.lag_seconds).toFixed(0) + "초"),
  );
  text(
    "learningDatabaseLocation",
    (m.replay_database_path || "경로 확인 중") +
      " · " +
      (Number(m.replay_file_bytes || 0) / 1048576).toFixed(2) +
      " MiB",
  );
  html("dailyLearningRows", () =>
    rows.length
      ? tableRows(
          rows.map((day) => [
            day.day,
            whole(day.enqueued),
            whole(day.first_trained),
            whole(day.completed),
            whole(day.remaining),
          ]),
        )
      : '<tr><td colspan="5">경험 생성 대기</td></tr>',
  );
  const bytes = Number(m.replay_file_bytes || 0),
    blocked =
      Number(m.replay_quarantined_count || 0) +
      Number(m.replay_unsupported_count || 0),
    forwards = m.candidate_training
      ? m.candidate_window_forwards_current
      : m.last_candidate_window_forwards;
  text(
    "dailyLearningSummary",
    "학습 가능한 남은 경험 " +
      whole(m.replay_eligible_backlog) +
      "건 · 양쪽 학습 시작 미완료 " +
      whole(m.replay_untrained_count) +
      "건 · 학습 불가·보류 " +
      whole(blocked) +
      "건 · replay " +
      (bytes / 1048576).toFixed(2) +
      " MiB · 용량 때문에 삭제하지 않음 · 최근 공유 입력 계산 " +
      whole(forwards) +
      "회",
  );
  if (m.replay_oldest_unfinished_timestamp)
    text(
      "dailyLearningSummary",
      textOf("dailyLearningSummary") +
        " · 가장 오래된 남은 경험 " +
        timeOf(m.replay_oldest_unfinished_timestamp),
    );
}

function renderDualLearning(d) {
  const m = d.metrics || {};
  const activeRole = m.champion_training
      ? "champion"
      : m.candidate_training
        ? "candidate"
        : null,
    backlog = Number(m.replay_eligible_backlog || 0),
    passes = m.candidate_replay_passes,
    learningOn = d.learning_enabled !== false;
  const status = !d.running
    ? "시스템 정지 · 저장된 마지막 학습 상태"
    : !learningOn
      ? "replay 학습 중지 · 미학습 " + whole(backlog) + "건 보존"
      : activeRole
        ? (activeRole === "champion" ? "Champion" : "Candidate") +
          " 학습 중 · 다른 모델은 다음 회차 대기"
        : backlog
          ? "학습 가능한 경험 " + whole(backlog) + "건 · 다음 회차 준비"
          : "학습 가능한 미학습 0건 · 새 결과 경험 대기";
  text("dualLearningStatus", status);
  badge(
    "learningState",
    !d.running
      ? "정지"
      : !learningOn
        ? "학습 OFF"
        : activeRole
          ? "학습 중"
          : backlog
            ? "회차 준비"
            : "새 경험 대기",
    !d.running ? "bad" : activeRole ? "blue" : "",
  );
  badge(
    "candidateBadge",
    !d.running
      ? "정지"
      : !learningOn
        ? "학습 OFF · 경험 보존"
        : activeRole
          ? "두 모델 순차 학습"
          : backlog
            ? "학습 회차 준비"
            : m.candidate_validation_active && d.observe_enabled === false
              ? "승급전 일시정지 · 판단 OFF"
              : m.candidate_validation_active
                ? "미학습 0 · 대결 진행"
                : "새 경험 대기",
    activeRole ? "blue" : "",
  );
  const health = d.agent_health || {},
    candidateHealth = health.candidate || {},
    lag = (value) =>
      value == null
        ? !d.agent_process_running
          ? "정지"
          : "미측정"
        : num(value).toFixed(0) + "초";
  text(
    "learningFlowHealth",
    status +
      (activeRole
        ? " · optimizer " +
          whole(m[activeRole + "_optimizer_steps_current"]) +
          " / " +
          whole(m[activeRole + "_optimizer_steps_target"]) +
          "회"
        : "") +
      " | 평가 대기 " +
      whole(m.replay_pending_count ?? m.pending_experiences) +
      "건 | 대결 " +
      whole(m.candidate_validation_bars) +
      " / " +
      whole(m.candidate_min_validation_dates || 390) +
      "개 시장 분 | Champion 지연 " +
      lag(health.lag_seconds) +
      " · Candidate 지연 " +
      lag(candidateHealth.lag_seconds),
  );
  for (const role of ["champion", "candidate"]) {
    const title = role === "champion" ? "Champion" : "Candidate",
      active = !!m[role + "_training"],
      prefix = "dual" + title;
    text(
      prefix + "State",
      !d.running
        ? "정지"
        : !learningOn
          ? "학습 OFF · " +
            whole(m[role + "_eligible_replay_count"] || 0) +
            "건 보존"
          : active
            ? "학습 중"
            : Number(m[role + "_eligible_replay_count"])
              ? "학습 회차 대기"
              : "남은 경험 완료 · 새 경험 대기",
    );
    text(
      prefix + "Version",
      whole(
        m[
          role === "champion"
            ? "champion_training_version"
            : "candidate_model_version"
        ],
      ),
    );
    text(prefix + "Runs", whole(m[role + "_completed_training_runs"]) + "회");
    text(
      prefix + "Exposures",
      whole(
        role === "champion"
          ? m.champion_paper_examples_trained
          : m.paper_examples_trained,
      ) + "개 paper 경험",
    );
    text(
      prefix + "Samples",
      whole(
        m[
          active
            ? role + "_samples_current"
            : "last_" + role + "_samples_trained"
        ],
      ) +
        " / " +
        whole(m[role + "_samples_target"]) +
        "개",
    );
    text(
      prefix + "Steps",
      whole(
        m[
          active
            ? role + "_optimizer_steps_current"
            : "last_" + role + "_optimizer_steps"
        ],
      ) +
        " / " +
        whole(m[role + "_optimizer_steps_target"]) +
        "회",
    );
    const completed = m[role + "_last_completed_round"];
    text(prefix + "CompletedAt", timeOf(completed?.completed_utc));
    text(
      prefix + "Time",
      completed
        ? num(completed.total_seconds).toFixed(1) +
            " / " +
            num(completed.compute_seconds).toFixed(1) +
            "초"
        : "완료 회차 측정 대기",
    );
    const vram = completed?.peak_allocated_bytes;
    text(
      prefix + "Vram",
      vram == null
        ? "측정 대기"
        : (Number(vram) / 1073741824).toFixed(2) + " GiB",
    );
    text(
      prefix + "Remaining",
      whole(m[role + "_eligible_replay_count"]) + "건",
    );
  }
  text(
    "compareChampionTrainable",
    whole(m.champion_trainable_parameter_count) + "개",
  );
  const policy =
    passes == null
      ? "학습 반복 횟수 미확인"
      : "Champion과 Candidate가 같은 경험을 각각 " +
        whole(passes) +
        "회 학습하고 저장합니다. 둘 다 끝난 경험만 삭제합니다.";
  text("learningThreshold", policy);
  text(
    "singlePassPolicy",
    policy +
      " 같은 GPU에서 회차를 번갈아 처리하며 서로 다른 시각의 경험과 모든 시간봉 입력은 유지합니다.",
  );
  const candidateRound = m.candidate_last_completed_round;
  if (candidateRound)
    text(
      "candidateTrainingTime",
      "최근 완료: 총 " +
        num(candidateRound.total_seconds).toFixed(1) +
        "초 · 계산 " +
        num(candidateRound.compute_seconds).toFixed(1) +
        "초 · 단계당 " +
        num(candidateRound.step_compute_seconds).toFixed(1) +
        "초",
    );
  text(
    "timeframeMeasurementAt",
    "입력 확보율 기준: " +
      (m.multiscale_input_status_utc
        ? timeOf(m.multiscale_input_status_utc)
        : "미측정"),
  );
  const timeframeRows = [];
  text(
    "modelUniverseSummary",
    "등록 " +
      whole(d.configured_instruments) +
      "종목 · 최근 5분 수신 " +
      whole((d.feed_metrics?.fresh_symbols_5m || []).length) +
      "종목 · 실제 판단 입력 " +
      (m.model_input_symbol_count == null
        ? "측정 대기"
        : whole(m.model_input_symbol_count) + "종목") +
      " · 빈 입력 칸 " +
      whole(m.model_padding_symbol_count) +
      "개 · 종목 ID 미연결 " +
      whole(m.unmatched_live_symbol_count) +
      "종목 · 모델 종목 ID 공간 " +
      (m.model_symbol_id_capacity == null
        ? "미확인"
        : whole(m.model_symbol_id_capacity) + "개") +
      " (동시 입력 가능 수 보장 아님)",
  );
  const names = {
    "1m": "1분봉",
    "3m": "3분봉",
    "5m": "5분봉",
    "15m": "15분봉",
    "60m": "60분봉",
    "1d": "일봉",
    "1w": "주봉",
    "1mo": "월봉",
  };
  for (const [scale, label] of Object.entries(names)) {
    const info = m.multiscale_input_status?.[scale],
      coverage = info?.mean_history_coverage ?? m.multiscale_coverage?.[scale],
      c = m.champion_last_completed_round?.timeframe_samples?.[scale],
      a = m.candidate_last_completed_round?.timeframe_samples?.[scale];
    const values = [
      label,
      coverage == null
        ? "측정 대기"
        : (Number(coverage) * 100).toFixed(1) + "%",
      info
        ? whole(info.available_symbols) + " / " + whole(info.observed_symbols)
        : "측정 대기",
      info ? whole(info.complete_history_symbols) + "종목" : "측정 대기",
      c == null
        ? "측정 대기"
        : whole(c) +
          " / " +
          whole(m.champion_last_completed_round.samples) +
          "개",
      a == null
        ? "측정 대기"
        : whole(a) +
          " / " +
          whole(m.candidate_last_completed_round.samples) +
          "개",
    ];
    timeframeRows.push(values);
  }
  html("timeframeLearningRows", () => tableRows(timeframeRows));

  text(
    "replayDetail",
    "학습 가능 " +
      whole(m.replay_eligible_backlog) +
      "건 · 보류 " +
      whole(
        Number(m.replay_quarantined_count || 0) +
          Number(m.replay_unsupported_count || 0),
      ) +
      "건 · 현재 DB 잔여",
  );
  const inputError =
    m.agent_last_input_error ||
    m.last_champion_error ||
    m.last_candidate_error ||
    m.candidate_validation_error;
  if (inputError)
    text(
      "learningFlowHealth",
      textOf("learningFlowHealth") + " | 오류: " + inputError,
    );
  const champError = m.last_champion_error;
  if (champError)
    text(
      "dualLearningStatus",
      textOf("dualLearningStatus") + " · Champion 오류: " + champError,
    );
}

function renderDailyOperation(d) {
  const input = d.input_availability || {},
    stored = input.stored || {},
    fresh = input.fresh_quotes || {},
    cycle = d.daily_cycle || {},
    v = d.validation_comparison || {},
    m = d.metrics || {};
  text(
    "inputQuickSummary",
    "등록 " +
      whole(input.configured) +
      " · 최근 수신 " +
      whole(input.fresh) +
      " · 모델 입력 " +
      (input.model_input == null ? "확인 대기" : whole(input.model_input)) +
      " · 설정상 매매 가능 " +
      whole(input.configured_tradable) +
      " · 문맥 전용 " +
      whole(input.context_only) +
      " · 호가 " +
      whole(fresh.bid_ask_symbols) +
      " · 잔량 " +
      whole(fresh.book_size_symbols) +
      " · Champion feed 대비 처리 커서 차이 " +
      (d.agent_health?.lag_seconds == null
        ? "미측정"
        : whole(d.agent_health.lag_seconds) + "초") +
      " · Candidate feed 대비 처리 커서 차이 " +
      (d.agent_health?.candidate?.lag_seconds == null
        ? "미측정"
        : whole(d.agent_health.candidate.lag_seconds) + "초"),
  );
  text(
    "dailyCycleSummary",
    "하루 승급전 판정 → 다음 고정 시험 준비 · 다음 " +
      timeOf(cycle.next_reset_utc) +
      " KST · 장기 운용 계좌·모델·미학습 경험 유지" +
      (cycle.in_progress ? " · 처리 중" : "") +
      " · 마지막 수동 초기화 " +
      timeOf(cycle.last_reset_utc),
  );
  const common = m.shared_observation || {},
    origins = common.experience_origins || {},
    hist = m.daily_history_input_status || {};
  const candidateHealth = d.agent_health?.candidate || {};
  const observationSource =
    common.source === "live_db"
      ? "replay DB 실시간 조회"
      : common.source === "replay_db_missing"
        ? "replay DB 없음"
        : "마지막 저장값 · 갱신 " +
          timeOf(common.updated_utc) +
          " · DB 조회 오류 " +
          (common.read_error || "확인");
  text(
    "sharedObservationSummary",
    observationSource +
      " · 공통 관찰 DB 누계 " +
      whole(common.common) +
      "회 · Champion 실제 전체 추론 누계 " +
      whole(m.champion_live_inference_count) +
      "회 · Candidate 실제 전체 추론 누계 " +
      whole(m.candidate_live_inference_count) +
      "회 · Candidate 큐 처리 완료 " +
      whole(common.candidate_completed) +
      "회 · 큐 대기 " +
      whole(common.pending) +
      "건 · 누계 기준 시작 점이 달라 직접 비교할 수 없습니다. 최신 시세 대비 Candidate 처리 커서 차이 " +
      (candidateHealth.lag_seconds == null
        ? "미측정"
        : whole(candidateHealth.lag_seconds) + "초") +
      (candidateHealth.status === "stale" || candidateHealth.status === "error"
        ? " · 확인 필요: " + candidateHealth.reason
        : "") +
      ". 커서 차이는 추론 소요시간이 아니며, 실제 Candidate 마지막 전체 추론 시각은 계좌 카드에 따로 표시합니다. 자체 경험 기록: Champion " +
      whole(origins.champion) +
      " / Candidate " +
      whole(origins.candidate) +
      ". 같은 행동이어도 서로 다른 계좌 결과는 별도 경험입니다.",
  );
  text(
    "dailyHistorySummary",
    "5년 이상 완료 일봉 확보 " +
      whole(hist.five_year_symbols) +
      " / " +
      whole(hist.observed_symbols) +
      "종목 · 일봉 없음 " +
      whole(hist.missing_symbols) +
      " · 인코더 최대 " +
      whole(hist.max_bars || 1300) +
      "개 일봉. 최근 실제 장기 일봉 포함 학습: Champion " +
      whole(m.champion_last_completed_round?.daily_history_samples) +
      " / Candidate " +
      whole(m.candidate_last_completed_round?.daily_history_samples) +
      "개 경험. 확보율은 마지막 전체 추론 " +
      (m.daily_history_input_status_utc
        ? timeOf(m.daily_history_input_status_utc)
        : "미측정") +
      " 기준입니다. 부족분은 공급자 수집으로 보충 중이며 없는 과거를 학습했다고 세지 않습니다.",
  );
  text(
    "dailyHistoryMeasurementAt",
    "입력 확보율 기준: " +
      (m.daily_history_input_status_utc
        ? timeOf(m.daily_history_input_status_utc)
        : "미측정"),
  );
  if ($("dailyHistoryCoverage").closest("details")?.open)
    html("dailyHistoryCoverage", () =>
      (hist.symbols || []).length
        ? accountTable(
            ["종목", "완료 일봉 수", "입력 기간", "5년 확보"],
            hist.symbols.map((row) => [
              instrumentLabel(row.symbol, d),
              whole(row.bars) + " / " + whole(hist.max_bars || 1300),
              decimal(row.years, 2) + "년",
              row.five_years_available
                ? "확보"
                : "부족 / 보충 또는 상장 기간 확인",
            ]),
          )
        : "일봉 인코더 입력 측정 대기",
    );
  const quoteRows = [
    ["OHLCV", "ohlcv_symbols"],
    ["최우선 매수·매도 호가 / 스프레드", "bid_ask_symbols"],
    ["매수·매도 잔량 / 잔량 불균형", "book_size_symbols"],
    ["매수·매도 체결량 / 체결 불균형", "directional_volume_symbols"],
    ["체결 횟수", "trade_count_symbols"],
  ].map(([name, key]) => [
    name,
    whole(stored[key]) + " / " + whole(input.configured),
    whole(fresh[key]) + " / " + whole(input.configured),
  ]);
  quoteRows.push([
    "1초 / 15초 / 30초 입력",
    "미수집 / 미수집 / 미수집",
    "미수집 / 미수집 / 미수집",
  ]);
  if ($("microstructureAvailability").closest("details")?.open)
    html("microstructureAvailability", () =>
      accountTable(
        ["입력 항목", "저장된 마지막 값", "최근 5분 갱신"],
        quoteRows,
      ),
    );
  const longRows = Object.entries(m.long_context_input_status || {}).map(
    ([name, info]) => [
      name,
      accountPercent(info.mean_history_coverage),
      whole(info.available_symbols) + " / " + whole(info.observed_symbols),
      m.champion_last_completed_round?.long_context_samples?.[name] == null
        ? "측정 대기"
        : whole(m.champion_last_completed_round.long_context_samples[name]),
      m.candidate_last_completed_round?.long_context_samples?.[name] == null
        ? "측정 대기"
        : whole(m.candidate_last_completed_round.long_context_samples[name]),
    ],
  );
  if ($("longContextAvailability").closest("details")?.open)
    html("longContextAvailability", () =>
      longRows.length
        ? accountTable(
            [
              "문맥 길이",
              "평균 확보율",
              "가용 종목",
              "Champion 최근 학습 경험",
              "Candidate 최근 학습 경험",
            ],
            longRows,
          )
        : "장기 문맥 측정 대기",
    );
  const before = d.universe_expansion?.before || {},
    obs = d.account_observability || {},
    gpu = d.physical_gpu || {},
    latest = m.champion_last_completed_round || {},
    previous = before.champion_round || {};
  const timing = (value) =>
      value == null ? "미측정" : decimal(value, 2) + "초",
    memory = (value) =>
      value == null ? "미측정" : decimal(value / 1024 ** 3, 2) + "GiB";
  if ($("universeCapacityComparison").closest("details")?.open)
    html("universeCapacityComparison", () =>
      accountTable(
        ["측정 항목", "확대 직전 · " + timeOf(before.measured_utc), "현재"],
        [
          [
            "등록 / 최근 수신 / 모델 입력",
            whole(before.configured) +
              " / " +
              whole(before.fresh) +
              " / " +
              whole(before.model_input),
            whole(input.configured) +
              " / " +
              whole(input.fresh) +
              " / " +
              whole(input.model_input),
          ],
          [
            "미학습 경험",
            whole(before.backlog),
            whole(m.replay_eligible_backlog),
          ],
          [
            "시세와 판단 지연",
            timing(before.agent_health?.lag_seconds),
            timing(d.agent_health?.lag_seconds),
          ],
          [
            "Champion 판단 시간",
            timing(before.champion_inference_seconds),
            timing(obs.champion?.last_inference_seconds),
          ],
          [
            "Candidate 판단 시간",
            timing(before.candidate_inference_seconds),
            timing(obs.candidate?.last_inference_seconds),
          ],
          [
            "Champion 최근 학습 경험 / optimizer 횟수",
            whole(previous.unique_samples) +
              " / " +
              whole(previous.optimizer_steps),
            whole(latest.unique_samples) +
              " / " +
              whole(latest.optimizer_steps),
          ],
          [
            "Champion 실제 학습 계산 / 총시간",
            timing(previous.compute_seconds) +
              " / " +
              timing(previous.total_seconds),
            timing(latest.compute_seconds) +
              " / " +
              timing(latest.total_seconds),
          ],
          [
            "학습 회차 최대 VRAM",
            memory(previous.peak_allocated_bytes),
            memory(latest.peak_allocated_bytes),
          ],
          [
            "장치 GPU 사용률",
            before.physical_gpu?.utilization_percent == null
              ? "미측정"
              : whole(before.physical_gpu.utilization_percent) + "%",
            gpu.utilization_percent == null
              ? "미측정"
              : whole(gpu.utilization_percent) + "%",
          ],
          [
            "최근 학습 회차 완료 시각",
            timeOf(previous.completed_utc),
            timeOf(latest.completed_utc),
          ],
        ],
      ),
    );
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
  const daily = (cycle.history || [])
    .slice()
    .reverse()
    .flatMap((row) =>
      ["champion", "candidate"].flatMap((role) =>
        Object.entries(row.accounts?.[role] || {}).map(([c, b]) => [
          row.session,
          role,
          c,
          accountPercent(b.net_return_rate),
          accountMoney(b.net_pnl, c),
          accountMoney(b.costs, c),
          whole(b.trade_count),
        ]),
      ),
    );
  if ($("dailyAccountHistory").closest("details")?.open)
    html("dailyAccountHistory", () =>
      daily.length
        ? accountTable(
            [
              "기록일",
              "모델",
              "통화",
              "계좌 누적 수익률",
              "누적 순손익",
              "누적 비용",
              "누적 체결",
            ],
            daily,
          )
        : "첫 일일 판정 기록 대기 중입니다. 장기 계좌 누적 기록과 고정 시험 승부 점수는 구분합니다.",
    );
}

function renderInferenceWork(d) {
  const m = d.metrics || {},
    obs = d.account_observability || {},
    p = obs.candidate?.inference_profile || {},
    o = m.candidate_live_observation_profile || {},
    seconds = (x) => (x == null ? "미측정" : decimal(x, 3) + "초");
  text(
    "inferenceWorkSummary",
    "Champion과 Candidate가 각자 추론·가상매매하고 양쪽 경험으로 각각 학습합니다. Champion " +
      whole(obs.champion?.inference_count) +
      "회 / Candidate " +
      whole(obs.candidate?.inference_count) +
      "회 · Candidate 미처리 관찰 " +
      whole(m.shared_observation?.pending) +
      "개. 학습 작업본과 승급전 고정본은 운영 계좌의 추론 모델과 구분합니다.",
  );
  const rows = [
    [
      "Champion 실시간 판단",
      seconds(obs.champion?.last_inference_seconds),
      "입력 준비 후 forward · GPU 완료 대기 포함",
    ],
    [
      "Candidate 관찰 처리 전체",
      seconds(o.total_seconds),
      "DB 읽기 → 체결·보상 → 추론 → 경험 저장 → 계좌 저장",
    ],
    [
      "Candidate DB 관찰 읽기",
      seconds(o.read_seconds),
      "저장소 잠금 대기 포함",
    ],
    [
      "Candidate 체결·보상 처리",
      seconds(o.fills_and_rewards_seconds),
      "성숙 경험을 한 transaction으로 기록",
    ],
    [
      "Candidate 경험·계좌 DB 기록",
      seconds(o.database_commit_seconds),
      "관찰 완료와 pending 경험을 함께 저장",
    ],
    [
      "Candidate 추론 경로 합계",
      seconds(p.total_seconds),
      "모델 잠금·GPU 대기 → forward → RAM 반환",
    ],
    [
      "Candidate 추론 대기",
      seconds(p.wait_seconds),
      "GPU 우선순위 대기 + 가중치 잠금 대기",
    ],
    ["Candidate 모델 GPU 이동", seconds(p.upload_seconds), "RAM → GPU"],
    [
      "Candidate 입력·forward",
      seconds(p.forward_seconds),
      "입력 준비 + forward + GPU 완료 대기",
    ],
    [
      "Candidate 모델 RAM 반환",
      seconds(p.download_seconds),
      "GPU → RAM · VRAM 확보",
    ],
    [
      "Candidate 관찰 가중치 갱신",
      seconds(m.candidate_observer_publish_seconds),
      m.candidate_observer_storage_reused
        ? "기존 모델 저장공간 재사용"
        : "최초 생성 / 측정 대기",
    ],
  ];
  for (const role of ["champion", "candidate"]) {
    const r = m[role + "_last_completed_round"] || {};
    rows.push([
      role === "champion" ? "Champion 최근 학습" : "Candidate 최근 학습",
      seconds(r.compute_seconds) + " / 총 " + seconds(r.total_seconds),
      whole(r.unique_samples) +
        "개 경험 · optimizer " +
        whole(r.optimizer_steps) +
        "회",
    ]);
  }
  html("inferenceWorkDetails", () =>
    accountTable(["작업", "최근 경과 시간", "측정 범위 / 처리량"], rows),
  );
}

function renderRuntimeUpdates(d) {
  let panel = document.getElementById("runtimeUpdatesPanel");
  if (!panel) {
    panel = document.createElement("details");
    panel.id = "runtimeUpdatesPanel";
    panel.className = "details";
    const title = document.createElement("summary");
    title.textContent = "설정·학습 코드 적용 상태 · 재시작 없이 갱신";
    panel.appendChild(title);
    const body = document.createElement("p");
    body.id = "runtimeUpdatesBody";
    body.style.whiteSpace = "pre-wrap";
    panel.appendChild(body);
    document.getElementById("runtimeDetails").appendChild(panel);
  }
  const m = d.metrics || {},
    u = m.runtime_updates,
    r = u && u.applied_rules;
  const lines = [
    "화면(JS/CSS): 새로고침 · 학습 규칙·loss/learner: 현재 회차 저장 후 hot apply",
    "모델 판단·관찰 Python: ‘모델 코드 적용’ 버튼으로 모델만 저장 후 재시작 · 시세 수집/웹 유지",
    "웹 API/server Python: 웹서버 재시작만 · feed/agent 유지",
  ];
  if (r) {
    lines.push(
      `현재 적용: batch ${r.training_batch_size} × optimizer ${r.training_optimizer_steps} · 결과 연결 ${r.reward_credit_seconds}초`,
    );
    lines.push(
      `GPU 학습: ${m.champion_optimizer_backend || "확인 중"} / ${m.candidate_optimizer_backend || "확인 중"} · 손실 계산 ${m.candidate_loss_backend || m.champion_loss_backend || "확인 중"}`,
    );
    if (u.applied_utc)
      lines.push(
        "최근 적용: " +
          new Date(u.applied_utc).toLocaleString("ko-KR", {
            timeZone: "Asia/Seoul",
          }),
      );
    const pending = Object.keys(u.deferred_rules || {});
    if (pending.length)
      lines.push(
        "보류: " +
          pending.join(", ") +
          " · 진행 중 승급전 조건과 활성 계좌 목표·입력 구조는 별도 적용 필요",
      );
    if (u.error) lines.push("적용 실패 · 기존 정상 값 유지: " + u.error);
  } else lines.push("실행 중 agent의 적용 상태를 확인 중입니다.");
  text("runtimeUpdatesBody", lines.join("\n"));
}
