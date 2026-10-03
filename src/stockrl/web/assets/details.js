"use strict";

// Expandable diagnostics: input coverage, GPU scheduling and runtime records.

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

function renderGpuScheduler(d) {
  const m = d.metrics || {},
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
    "opGpuWork",
    !runningModelRoles(d).length
      ? "정지"
      : !d.agent_process_running
        ? "TradingMoE 전용 작업 · 기존 작업 대기열과 별개"
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
}

function renderAccountInputContext(d) {
  const view = d.account_observability;
  if (!view) {
    text("accountInputScope", "계좌 입력 측정 없음");
    text("accountRewardHorizon", "보상 연결 측정 없음");
    return;
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

function renderProcessingHistory(d) {
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
  text("feeTotal", decimal(m.fee_total, 4));
  text("slippageTotal", decimal(m.slippage_total, 4));
  property(
    "paperReward",
    "className",
    num(m.paper_net_reward) < 0 ? "value-negative" : "value-positive",
  );
  text(
    "paperReward",
    m.paper_net_reward == null
      ? "—"
      : (Number(m.paper_net_reward) * 100).toFixed(3) + "% (비용 차감)",
  );
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
}
