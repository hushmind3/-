"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = vm.createContext({
  num: (value) => (Number.isFinite(Number(value)) ? Number(value) : 0),
  whole: (value) => Math.round(Number(value) || 0).toLocaleString("ko-KR"),
  timeOf: (value) => value,
});
for (const screen of ["dashboard", "learning", "trial"])
  vm.runInContext(
    fs.readFileSync("src/stockrl/web/assets/" + screen + ".js", "utf8"),
    context,
  );
const base = {
  agent_process_running: true,
  learning_enabled: true,
  metrics: {
    replay_eligible_backlog: 0,
    replay_pending_count: 1944,
    champion_last_completed_round: {
      completed_utc: "2026-10-01T19:33:36Z",
      samples: 92,
    },
  },
};
const cases = [
  [
    "no completed history",
    {
      ...base,
      metrics: { replay_eligible_backlog: 0, replay_pending_count: 10 },
    },
    /^학습 ON.*손익 확인 중/,
  ],
  [
    "disconnected",
    { ...base, status_unavailable: true },
    /서버 연결 끊김/,
    /현재 학습 여부.*확인할 수 없/,
    /5초/,
  ],
  [
    "completed, waiting results",
    base,
    /모두 학습 완료.*결과 대기/,
    /1,944/,
    /자동.*학습/,
  ],
  [
    "completed, waiting new",
    { ...base, metrics: { ...base.metrics, replay_pending_count: 0 } },
    /새 경험 대기/,
  ],
  ["off", { ...base, learning_enabled: false }, /학습 OFF/, /DB/, /ON/],
  ["stopped", { ...base, agent_process_running: false }, /프로세스 정지/],
  [
    "unknown process",
    { ...base, agent_process_running: undefined },
    /상태 미확인/,
  ],
  ["unknown switch", { ...base, learning_enabled: undefined }, /설정 미확인/],
  ["unknown count", { ...base, metrics: {} }, /학습량 미확인/],
  [
    "candidate active",
    { ...base, metrics: { ...base.metrics, candidate_training: true } },
    /Candidate 학습 중/,
  ],
  [
    "both active",
    {
      ...base,
      metrics: {
        ...base.metrics,
        candidate_training: true,
        champion_training: true,
      },
    },
    /Champion.*Candidate.*학습 중/,
  ],
  [
    "active GPU wait",
    {
      ...base,
      metrics: {
        ...base.metrics,
        candidate_training: true,
        learning_wait_reason: "실시간 GPU 추론 요청에 양보",
      },
    },
    /학습 회차 진행.*연산 대기/,
    /GPU/,
  ],
  [
    "queued GPU wait",
    {
      ...base,
      metrics: {
        ...base.metrics,
        replay_eligible_backlog: 120,
        learning_wait_reason: "실시간 GPU 추론 요청에 양보",
      },
    },
    /처리 순서 대기/,
    /120/,
  ],
  [
    "ready",
    { ...base, metrics: { ...base.metrics, replay_eligible_backlog: 120 } },
    /다음 회차 준비/,
  ],
  [
    "blocked",
    { ...base, metrics: { ...base.metrics, replay_quarantined_count: 14 } },
    /보류 확인/,
    /14건/,
  ],
  [
    "error",
    {
      ...base,
      metrics: { ...base.metrics, last_candidate_error: "fixture error" },
    },
    /오류 확인/,
    /fixture error/,
  ],
];
for (const [name, data, title, reason, next] of cases) {
  const state = context.learningSituation(data);
  assert.match(state.title, title, name);
  if (reason) assert.match(state.reason, reason, name);
  if (next) assert.match(state.next, next, name);
  for (const key of ["title", "reason", "next"])
    assert.equal(typeof state[key], "string", name);
  // Every summary comes from one state computation, including errors and OFF.
  const values = new Map();
  context.text = (id, value) => values.set(id, value);
  context.property = () => {};
  context.badge = (id, value) => values.set(id, value);
  context.renderLearningSituation(data);
  assert.equal(values.get("learningState"), state.title, name);
  assert.equal(values.get("learningSituationTitle"), state.title, name);

  assert.equal(values.get("learningSituationReason"), state.reason, name);
}
console.log(
  JSON.stringify({
    learningStateCases: cases.length,
    matchingSummaryAndBadge: "passed",
  }),
);

const pendingCases = [
  [{ metrics: {} }, /미확인/, /미확인/],
  [{ metrics: { replay_pending_count: 0 } }, /미확인/, /대기 없음/],
  [
    {
      metrics: {
        replay_pending_count: 2062,
        reward_credit: { duration_seconds: 3600 },
        champion_pending_reward_status: { reasons: { next_quote: 1031 } },
        candidate_pending_reward_status: { reasons: { next_quote: 1031 } },
      },
    },
    /60분/,
    /새 시세 · 2,062건/,
  ],
  [
    {
      metrics: {
        replay_pending_count: 5,
        reward_credit: { duration_seconds: 900 },
        champion_pending_reward_status: {
          reasons: { reward_horizon: 3, fill: 2 },
        },
      },
    },
    /15분/,
    /손익 확인 시간 경과 · 3건.*가상 주문 체결 · 2건/,
  ],
  [
    {
      metrics: {
        replay_pending_count: 2,
        champion_pending_reward_status: {
          reasons: { missing_market_input: 1, blocked: 1 },
        },
      },
    },
    /미확인/,
    /시세 입력 복구.*기록 오류 해결/,
  ],
];
for (const [d, window, reason] of pendingCases) {
  const state = context.pendingOutcomeState(d);
  assert.match(state.window, window);
  assert.match(state.reason, reason);
}
const trialCases = [
  [
    {
      agent_process_running: true,
      observe_enabled: false,
      validation_comparison: { active: true },
    },
    /모델 판단 OFF/,
  ],
  [
    {
      agent_process_running: true,
      observe_enabled: true,
      validation_comparison: {
        active: true,
        bars_current: 69,
        bars_required: 390,
      },
      daily_cycle: { next_reset_utc: "07:00" },
    },
    /321개 남음.*07:00/,
  ],
  [
    {
      agent_process_running: true,
      validation_comparison: { status: "promoted", comparison_valid: true },
    },
    /판정 완료.*장기 가상계좌는 유지/,
  ],
  [{ agent_process_running: false }, /프로세스가 정지/],
  [{ status_unavailable: true }, /연결 끊김/],
];
for (const [d, re] of trialCases) assert.match(context.trialNextAction(d), re);
console.log(
  JSON.stringify({
    pendingExplanationCases: pendingCases.length,
    trialNextActionCases: trialCases.length,
  }),
);

assert.match(
  context.pendingOutcomeState({
    metrics: {
      replay_pending_count: 100,
      champion_pending_reward_status: { reasons: { next_quote: 90 } },
    },
  }).reason,
  /사유 집계 갱신 중 · 10건/,
);

const modelSource = fs.readFileSync("src/stockrl/multiscale.py", "utf8");
const modelLookbacks = [
  ...modelSource
    .match(/_LOOKBACK_BARS\s*=\s*\{([^}]+)\}/)[1]
    .matchAll(/"([^" ]+)":\s*(\d+)/g),
].map(([, name, n]) => [name, Number(n) + 1]);
const uiSource = fs.readFileSync("src/stockrl/web/assets/learning.js", "utf8");
for (const [name, n] of modelLookbacks)
  assert(
    new RegExp('"' + name + '":\\s*' + n + "[,\\s]").test(uiSource),
    name + " required completed bars match model",
  );
console.log(
  JSON.stringify({ timeframeRequirementChecks: modelLookbacks.length }),
);

const metricDay = new Intl.DateTimeFormat("sv-SE", {
  timeZone: "Asia/Seoul",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
}).format(new Date());
const liveCards = context.operatorMetrics({
  agent_process_running: true,
  learning_enabled: true,
  observe_enabled: true,
  paper_enabled: true,
  feed_running: true,
  configured_instruments: 200,
  feed_metrics: { fresh_symbols_5m: ["A", "B"] },
  agent_health: {
    lag_seconds: 0,
    threshold_seconds: 300,
    candidate: { lag_seconds: 60 },
  },
  metrics: {
    model_input_symbol_count: 100,
    replay_eligible_backlog: 25,
    replay_pending_count: 90,
    daily_learning: [{ day: metricDay, completed: 75, enqueued: 100 }],
  },
  paper_financials: { KRW: { trade_count: 10 }, USD: { trade_count: 20 } },
  candidate_live_account: {
    books: { KRW: { trade_count: 30 }, USD: { trade_count: 40 } },
  },
  validation_comparison: { active: true, bars_current: 10, bars_required: 100 },
});
assert.equal(liveCards.Feed.number, "2 / 200종목");
assert.equal(liveCards.Feed.ratio, 1);
assert.equal(liveCards.Inference.number, "60초");
assert.equal(liveCards.Inference.ratio, 20);
assert.equal(liveCards.Paper.number, "100건");
assert.match(liveCards.Paper.detail, /Champion 30건.*Candidate 70건/);
assert.equal(liveCards.Learning.number, "75 / 100건");
assert.equal(liveCards.Learning.ratio, 75);
assert.match(liveCards.Learning.detail, /남은 학습 25건.*손익 확인 90건/);
assert.equal(liveCards.Trial.ratio, 10);
for (const card of Object.values(
  context.operatorMetrics({ status_unavailable: true }),
)) {
  assert.equal(card.number, "—");
  assert.equal(card.status, "연결 끊김");
  assert.equal(card.ratio, null);
}
assert.equal(context.operatorMetrics({}).Learning.status, "미확인");
assert.equal(
  context.operatorMetrics({
    agent_process_running: true,
    learning_enabled: false,
  }).Learning.status,
  "OFF",
);
console.log(JSON.stringify({ metricCardChecks: 12 }));
