"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = vm.createContext({
  num: (value) => (Number.isFinite(Number(value)) ? Number(value) : 0),
  whole: (value) => Math.round(Number(value) || 0).toLocaleString("ko-KR"),
  timeOf: (value) => value,
});
vm.runInContext(
  fs.readFileSync("src/stockrl/web/assets/dashboard.js", "utf8"),
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
    /^학습 ON.*결과 평가 대기/,
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
  assert.match(values.get("learningAtGlance"), /학습 상태 보기/, name);
  assert.equal(values.get("learningSituationReason"), state.reason, name);
}
console.log(
  JSON.stringify({
    learningStateCases: cases.length,
    matchingSummaryAndBadge: "passed",
  }),
);
