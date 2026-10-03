"use strict";

// Experience learning: lifecycle, canonical activity, model rounds and replay.

function renderExperienceFlow(d) {
  const metrics = d.metrics || {};
  text("opLearningQueue", whole(metrics.replay_eligible_backlog) + "개");
  text("opLastLearning", "마지막 학습·저장 " + timeOf(metrics.last_update_utc));

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
  attribute("flowPendingWaitReason", "title", pendingState.reason);
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
  const roles = runningModelRoles(d);
  const legacyRoles = roles.filter((role) => d.model_runtime?.[role]?.memory_scope !== "worker");
  const workerRoles = roles.filter((role) => d.model_runtime?.[role]?.memory_scope === "worker");
  const error = d.agent_process_running
    ? m.learner_statistics_error || m.agent_last_input_error || legacyRoles.map((role) => m["last_" + role + "_error"]).find(Boolean)
    : null;
  const active = legacyRoles
    .filter((role) => m[role + "_training"])
    .map((role) => role === "champion" ? "Champion" : "Candidate")
    .join(" · ");
  if (workerRoles.length && !legacyRoles.length)
    return {
      title: workerRoles.map((role) => role === "champion" ? "Champion" : "Candidate").join(" · ") + (d.learning_enabled === false ? " 학습 중지" : " TradingMoE · 학습 허용"),
      reason: "공식 ETHUSDT 과거 구간을 실행합니다. 이 모델의 계좌와 replay는 기존 두 모델의 replay DB와 별개입니다.",
      next: "해당 모델의 상세에서 최근 학습 결과를 확인하세요. 아래 DB 수치는 기존 공통 replay입니다.",
      tone: d.learning_enabled === false ? "" : "blue",
    };
  const wait = String(m.learning_wait_reason || "");
  if (d.agent_process_running !== true)
    return {
      title:
        d.agent_process_running === false
          ? "학습 프로세스 정지"
          : "학습 프로세스 상태 미확인",
      reason: "현재 실행 중인 학습 프로세스가 확인되지 않습니다.",
      next: "운영 · 계좌의 Champion 시작 또는 Candidate 시작으로 모델을 실행하세요. 시스템 시작은 시세 수집만 켭니다.",
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
  badge("candidateBadge", state.title, state.tone);
  text("dualLearningStatus", state.title);

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
    const active = modelIsLearning(d, role);
    const remaining = m[role + "_eligible_replay_count"];
    const state = learningSituation(d);
    const lifecycle = d.model_runtime?.[role];
    const label = lifecycle && !modelIsRunning(d, role)
      ? modelStateLabel(lifecycle) + " · 기존 공통 학습 기록"
      : lifecycle?.memory_scope === "worker"
        ? "TradingMoE · " + (d.learning_enabled === false ? "학습 중지" : "학습 허용") + " · 별도 replay 사용"
      : d.agent_process_running !== true
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
    const lastRound = m[role + "_last_completed_round"];
    text(
      "learner" + role + "Samples",
      lastRound?.samples == null ? "—" : whole(lastRound.samples) + "건",
    );
    text(
      "learner" + role + "Seconds",
      lastRound?.total_seconds == null
        ? "—"
        : decimal(lastRound.total_seconds, 1) + "초",
    );
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

function renderLearningTotals(d) {
  const m = d.metrics || {};

  const training = modelIsLearning(d, "candidate");

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
  const l = d.learning || {};
  if (!l.promotion_gate_ready && !m.last_promotion_utc && !m.last_rejection_utc)
    text("lastGateResult", "비교 전 | 승급 차단");

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

function renderModelComparison(d) {
  if (!detailsOpen("compareChampionParams")) return;
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
      training = modelIsLearning(d, role);
    text(
      "compare" + title + "LearningState",
      !modelIsRunning(d, role)
        ? "정지 · 마지막 기록"
        : d.model_runtime?.[role]?.memory_scope === "worker"
          ? "TradingMoE · 별도 학습 기록 사용"
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
  text(
    "modelRuntimeParameterCount",
    d.metrics?.parameters == null
      ? "모델 대기 중"
      : whole(d.metrics.parameters) + "개",
  );

  renderModelComparison(d);
  const m = d.metrics || {},
    l = d.learning || {},
    gb = (v) => (v == null ? "—" : (Number(v) / 1073741824).toFixed(2) + " GB"),
    pct = (v) => (v == null ? "—" : (Number(v) * 100).toFixed(4) + "%"),
    missing = "아직 측정 전";
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
    training = modelIsLearning(d, "candidate"),
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
      : modelIsLearning(d, "candidate")
        ? "학습 중 · 이번 회차 완료 후 시간 측정"
        : "아직 측정 전",
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
  const removed = rows.reduce(
    (n, day) => n + num(day.removed_without_completion),
    0,
  );
  text(
    "removedExperienceNote",
    removed
      ? "기록 기간 중 학습 완료로 집계하지 않고 별도 제거한 경험 " +
          whole(removed) +
          "건. 현재 DB의 미학습 잔여에는 포함하지 않습니다."
      : "",
  );
  const bytes = Number(m.replay_file_bytes || 0),
    blocked =
      Number(m.replay_quarantined_count || 0) +
      Number(m.replay_unsupported_count || 0),
    forwards = modelIsLearning(d, "candidate")
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
  const activeRole = modelIsLearning(d, "champion")
      ? "champion"
      : modelIsLearning(d, "candidate")
        ? "candidate"
        : null,
    backlog = Number(m.replay_eligible_backlog || 0),
    passes = m.candidate_replay_passes,
    learningOn = d.learning_enabled !== false;
  for (const role of ["champion", "candidate"]) {
    const title = role === "champion" ? "Champion" : "Candidate",
      active = modelIsLearning(d, role),
      prefix = "dual" + title;
    text(
      prefix + "State",
      !modelIsRunning(d, role)
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
  const requiredBars = {
    "1m": 129,
    "3m": 65,
    "5m": 65,
    "15m": 33,
    "60m": 5,
    "1d": 65,
    "1w": 53,
    "1mo": 25,
  };
  for (const [scale, label] of Object.entries(names)) {
    const info = m.multiscale_input_status?.[scale],
      coverage = info?.mean_history_coverage ?? m.multiscale_coverage?.[scale],
      c = m.champion_last_completed_round?.timeframe_samples?.[scale],
      a = m.candidate_last_completed_round?.timeframe_samples?.[scale];
    const values = [
      label,
      whole(requiredBars[scale]) + "개 완료 봉",
      coverage == null
        ? "측정 대기"
        : (Number(coverage) * 100).toFixed(2) + "%",
      info
        ? whole(info.available_symbols) + " / " + whole(info.observed_symbols)
        : "측정 대기",
      info
        ? "충분 " +
          whole(info.complete_history_symbols) +
          " / 부족 " +
          whole(
            Math.max(
              0,
              Number(info.observed_symbols) -
                Number(info.complete_history_symbols),
            ),
          ) +
          "종목"
        : "측정 대기",
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
  const incomplete = Object.entries(names)
    .filter(([scale]) => {
      const row = m.multiscale_input_status?.[scale];
      return (
        row &&
        Number(row.complete_history_symbols) < Number(row.observed_symbols)
      );
    })
    .map(([, label]) => label);
  text(
    "timeframeInputWarning",
    !m.multiscale_input_status
      ? "과거 기록 충분 여부 미확인"
      : incomplete.length
        ? "기록 부족: " +
          incomplete.join(" · ") +
          " — 확보된 일부 기록으로 판단·학습하고 있습니다. 아래 ‘최근 학습’ 건수는 입력 포함 여부이며, 과거 기록이 충분하다는 뜻은 아닙니다."
        : "모든 시간봉에서 필요한 과거 봉 수를 충족했습니다.",
  );
  property(
    "timeframeInputWarning",
    "className",
    "notice " + (incomplete.length ? "input-incomplete" : ""),
  );
}
