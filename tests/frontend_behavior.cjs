"use strict";
// All commands are mocked. This suite never starts/stops/resets the live agents.
const assert = require("node:assert/strict");
const { build, fixture } = require("./frontend_regression.cjs");
const { performance } = require("node:perf_hooks");
let checks = 0;
function check(task) {
  task();
  checks++;
}
const clone = (value) => structuredClone(value);
function route(obj, page) {
  obj.w.history.replaceState(null, "", "/#" + page);
  obj.w.selectDashboardPage();
}
function book(rate, pnl) {
  return {
    initial_cash: 10000,
    cash: 2000,
    equity: 10000 + pnl,
    net_pnl: pnl,
    net_return_rate: rate,
    cash_ratio: 0.2,
    largest_position_weight: 0.3,
    costs: 15,
    trade_count: 12,
    position_count: 2,
    positions: [
      {
        symbol: "AAPL",
        quantity: 2,
        mark: 200,
        average_cost: 180,
        unrealized_pnl: 40,
        weight: 0.04,
        mark_available: true,
      },
    ],
  };
}
const rich = {
  ...clone(fixture),
  paper_financials: { KRW: book(0.01, 100), USD: book(0.02, 200) },
  candidate_live_account: {
    books: { KRW: book(-0.01, -100), USD: book(0.03, 300) },
  },
  account_observability: {
    champion: { books: { KRW: book(0.01, 100), USD: book(0.02, 200) } },
    candidate: { books: { KRW: book(-0.01, -100), USD: book(0.03, 300) } },
  },
  metrics: {
    ...fixture.metrics,
    replay_file_bytes: 10485760,
    candidate_incompatible_replay_count: 14,
    daily_learning: [{ day: "2026-10-02", completed: 80, enqueued: 100 }],
    champion_last_completed_round: {
      samples: 512,
      total_seconds: 12.25,
      loss_mean: -0.012,
    },
    candidate_last_completed_round: {
      samples: 128,
      total_seconds: 5.5,
      loss_mean: 0.03,
    },
  },
};
async function main() {
  const obj = build(true, clone(rich));
  obj.w.render(rich);
  const errors = build(true, clone(rich));
  errors.w.render({
    ...rich,
    metrics: { ...rich.metrics, last_candidate_error: "fixture error" },
  });
  check(() =>
    assert.equal(
      errors.w.document.getElementById("runtimeAlertPanel").hidden,
      false,
    ),
  );
  check(() =>
    assert.match(
      errors.w.document.getElementById("runtimeAlertSummary").textContent,
      /오류·지연.*건/,
    ),
  );
  errors.w.document.getElementById("runtimeAlertPanel").open = true;
  errors.w.render({
    ...rich,
    metrics: { ...rich.metrics, last_candidate_error: "fixture error" },
  });
  check(() =>
    assert.match(
      errors.w.document.getElementById("runtimeAlert").textContent,
      /fixture error/,
    ),
  );
  errors.w.render(rich);
  check(() =>
    assert.equal(
      errors.w.document.getElementById("runtimeAlertPanel").hidden,
      true,
    ),
  );
  route(obj, "control");
  const doc = obj.w.document;
  check(() =>
    assert.equal(
      new Set([...doc.querySelectorAll("[id]")].map((n) => n.id)).size,
      doc.querySelectorAll("[id]").length,
    ),
  );
  check(() =>
    assert.equal(doc.getElementById("championKRWReturn").textContent, "1.00%"),
  );
  check(() =>
    assert.equal(doc.getElementById("candidateUSDReturn").textContent, "3.00%"),
  );
  check(() =>
    assert.equal(
      doc.getElementById("overviewReplaySize").textContent,
      "10.0 MiB",
    ),
  );
  check(() =>
    assert.equal(
      doc.getElementById("overviewExperienceCompleted").textContent,
      "80건",
    ),
  );
  const accountNode = doc.getElementById("championKRWReturn");
  obj.w.render({
    ...rich,
    account_observability: {
      ...rich.account_observability,
      champion: { books: { KRW: book(0.05, 500), USD: book(0.02, 200) } },
    },
  });
  check(() =>
    assert.equal(doc.getElementById("championKRWReturn"), accountNode),
  );
  check(() => assert.equal(accountNode.textContent, "5.00%"));
  // Any hidden screen renderer would throw. Refresh of overview must not call them.
  const spies = [
    "renderMarkets",
    "renderDecisions",
    "renderLearningTotals",
    "renderTrialSummary",
    "renderValidationAccounts",
    "renderInferenceWork",
    "renderConnection",
  ];
  const original = Object.fromEntries(spies.map((k) => [k, obj.w[k]]));
  for (const k of spies)
    obj.w[k] = () => {
      throw Error("Hidden screen rendered: " + k);
    };
  check(() => obj.w.render(rich));
  for (const k of spies) obj.w[k] = original[k];
  const panel = doc.getElementById("positions").closest("details");
  let positionReads = 0;
  const positions = new Proxy(
    rich.account_observability.champion.books.KRW.positions,
    {
      get(target, key) {
        positionReads++;
        return Reflect.get(target, key);
      },
    },
  );
  const expensive = clone(rich);
  expensive.account_observability.champion.books.KRW.positions = positions;
  obj.w.render(expensive);
  check(() => assert.equal(positionReads, 0));
  panel.open = true;
  obj.w.render(expensive);
  check(() => assert(positionReads > 0));
  check(() =>
    assert.match(doc.getElementById("positions").textContent, /AAPL|Apple/),
  );
  const nested = doc
    .getElementById("learnerchampionState")
    .closest("[data-view]");
  route(obj, "learning");
  check(() =>
    assert.equal(
      doc.getElementById("learnerchampionSamples").textContent,
      "512건",
    ),
  );
  check(() =>
    assert.equal(
      doc.getElementById("learnerchampionSeconds").textContent,
      "12.3초",
    ),
  );
  // Multiple folded ancestors suppress the expensive HTML callback.
  const outer = doc.createElement("details"),
    inner = doc.createElement("details"),
    target = doc.createElement("div");
  target.id = "nestedRenderProbe";
  nested.append(outer);
  outer.append(inner);
  inner.append(target);
  inner.open = true;
  let built = 0;
  obj.w.html("nestedRenderProbe", () => {
    built++;
    return "content";
  });
  check(() => assert.equal(built, 0));
  outer.open = true;
  obj.w.html("nestedRenderProbe", () => {
    built++;
    return "content";
  });
  check(() => assert.equal(built, 1));
  for (const page of [
    "control",
    "markets",
    "learning",
    "promotionTrial",
    "connection",
    "system",
  ]) {
    route(obj, page);
    check(() =>
      assert.equal(
        [...doc.querySelectorAll("[data-view]")].filter((n) => !n.hidden)
          .length,
        1,
      ),
    );
  }
  check(() => assert.deepEqual(obj.calls, []));
  obj.w.close();
  // Fresh direct entry into each page must not depend on opening overview first.
  for (const [page, id, pattern] of [
    ["learning", "modelRuntimeParameterCount", /12,345개/],
    ["learning", "opLearningQueue", /12개/],
    ["markets", "championVersion", /v-test/],
    ["markets", "outputDiagnostic", /출력 점검/],
    ["system", "accountInputScope", /요약을 함께 입력/],
    ["system", "replayCount", /8/],
    ["system", "paperReward", /1.000%/],
  ]) {
    const data = {
      ...clone(rich),
      champion_version: "v-test",
      metrics: {
        ...rich.metrics,
        parameters: 12345,
        replay_count: 8,
        paper_net_reward: 0.01,
      },
    };
    const fresh = build(true, data),
      field = fresh.w.document.getElementById(id);
    for (
      let parent = field.parentElement;
      parent;
      parent = parent.parentElement
    )
      if (parent.tagName === "DETAILS") parent.open = true;
    fresh.w.render(data);
    route(fresh, page);
    check(() => assert.match(field.textContent, pattern));
    fresh.w.close();
  }
  // POST failure restores the same settings and tells the user.
  for (const [id, key] of [
    ["observeBtn", "observe_enabled"],
    ["paperBtn", "paper_enabled"],
    ["learningBtn", "learning_enabled"],
  ]) {
    const state = clone(fixture),
      o = build(true, state);
    o.w.render(state);
    const btn = o.w.document.getElementById(id),
      before = btn.textContent;
    let release;
    o.w.fetch = async (url, opt = {}) =>
      opt.method === "POST"
        ? new Promise((resolve) => {
            release = () =>
              resolve({
                ok: false,
                json: async () => ({ error: "mock rejection" }),
              });
          })
        : { ok: true, json: async () => state };
    const waiting = btn.onclick();
    check(() => assert.match(btn.textContent, /요청 중/));
    check(() => assert.equal(btn.getAttribute("aria-busy"), "true"));
    o.w.render(state);
    check(() => assert.match(btn.textContent, /요청 중/));
    release();
    await waiting;
    check(() => assert.equal(btn.textContent, before));
    check(() => assert.equal(state[key], true));
    check(() =>
      assert.match(
        o.w.document.getElementById("commandFeedback").textContent,
        /mock rejection/,
      ),
    );
    o.w.close();
  }
  // Refuse unknown flags rather than guessing a setting; independent 24 combinations are in the regression suite.
  const unknown = build(true, {
    ...clone(fixture),
    learning_enabled: undefined,
  });
  unknown.w.render({ ...fixture, learning_enabled: undefined });
  await unknown.w.document.getElementById("learningBtn").onclick();
  check(() => assert.deepEqual(unknown.calls, []));
  unknown.w.close();
  // A cancelled reset has no POST. Accepting it calls only the original reset route.
  for (const accept of [false, true]) {
    const o = build(true, clone(fixture));
    o.w.render(fixture);
    o.w.confirm = () => accept;
    await o.w.document.getElementById("resetAccountsBtn").onclick();
    check(() =>
      assert.deepEqual(
        o.calls,
        accept ? [{ url: "/api/paper-accounts/reset", payload: {} }] : [],
      ),
    );
    o.w.close();
  }
  // All operator commands retain their endpoint and body (restart health endpoint is checked by the static contract).
  for (const [id, url, payload] of [
    ["applyBtn", "/api/start", { mode: "live" }],
    ["reconnectBtn", "/api/feed/reconnect", {}],
    ["agentReloadBtn", "/api/agent/reload", {}],
    ["stopBtn", "/api/stop", {}],
    [
      "recheckBtn",
      "/api/provider/test",
      { provider: "kiwoom", environment: "real" },
    ],
    [
      "connectBtn",
      "/api/provider/connect",
      {
        environment: "real",
        app_key: "test-key",
        secret: "test-secret",
        account: "test-account",
      },
    ],
  ]) {
    const o = build(true, clone(fixture));
    o.w.render(fixture);
    o.w.document.getElementById("appKey").value = "test-key";
    o.w.document.getElementById("secret").value = "test-secret";
    o.w.document.getElementById("account").value = "test-account";
    await o.w.document.getElementById(id).onclick();
    check(() => assert.deepEqual(o.calls, [{ url, payload }]));
    o.w.close();
  }
  // Coalesce overlapping refreshes; do not duplicate API work.
  const coalesced = build(true, clone(fixture));
  let gets = 0,
    release;
  coalesced.w.fetch = () => {
    gets++;
    return new Promise(
      (resolve) =>
        (release = () => resolve({ ok: true, json: async () => fixture })),
    );
  };
  const reads = [
    coalesced.w.refresh(),
    coalesced.w.refresh(),
    coalesced.w.refresh(),
  ];
  check(() => assert.equal(gets, 1));
  release();
  await Promise.all(reads);
  coalesced.w.close();
  // Compare work on the actual selected overview, not on six simultaneous unhidden test pages.
  const benchmark = [];
  for (const isNew of [false, true]) {
    const o = build(isNew, clone(rich));
    o.w.render(rich);
    route(o, "control");
    let hiddenCalls = 0;
    const saved = o.w.renderModelComparison;
    o.w.renderModelComparison = (...args) => {
      hiddenCalls++;
      return saved(...args);
    };
    const obs = new o.w.MutationObserver(() => {});
    obs.observe(o.w.document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      characterData: true,
    });
    const times = [];
    let mutations = 0;
    for (let i = 0; i < 20; i++) {
      const at = performance.now();
      o.w.render(rich);
      times.push(performance.now() - at);
      mutations += obs.takeRecords().length;
    }
    times.sort((a, b) => a - b);
    benchmark.push({
      version: isNew ? "after" : "before",
      medianMs: +times[10].toFixed(3),
      mutationsPerRefresh: mutations / 20,
      hiddenLearningRenderCalls: hiddenCalls,
    });
    obs.disconnect();
    o.w.close();
  }
  console.log(
    JSON.stringify(
      { behaviorChecks: checks, benchmark, liveCommandsSubmitted: 0 },
      null,
      2,
    ),
  );
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
