/** Frontend contract regression. No requests are sent to the trading server.
 * Install jsdom into a temporary tools directory and set NODE_PATH; see docs/frontend.md.
 * Optional --status-file uses an existing status JSON as an additional real-data fixture.
 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const cp = require("node:child_process");
const path = require("node:path");
const vm = require("node:vm");
const { performance } = require("node:perf_hooks");
const { JSDOM } = require("jsdom");
const baseline = "a0f21b7";
const assetRoot = "src/stockrl/web/assets/";
const oldNames = ["app", "dashboard", "details", "controls"];
const newNames = [
  "app",
  "dashboard",
  "details",
  "accounts",
  "learning",
  "trial",
  "experts",
  "controls",
];
const gitRead = (file) =>
  cp.execFileSync("git", ["show", `${baseline}:${file}`], {
    encoding: "utf8",
    maxBuffer: 2e6,
  });
const oldHtml = gitRead("src/stockrl/web_dashboard.html");
const newHtml = fs.readFileSync("src/stockrl/web_dashboard.html", "utf8");
const oldCode = oldNames.map((name) => gitRead(assetRoot + name + ".js"));
const newCode = newNames.map((name) =>
  fs.readFileSync(assetRoot + name + ".js", "utf8"),
);
const dateNow = Date.now();
const fixture = {
  running: true,
  agent_running: true,
  agent_process_running: true,
  feed_running: true,
  observe_enabled: true,
  paper_enabled: true,
  learning_enabled: true,
  configured_instruments: 2,
  metrics: {
    device: "cuda",
    replay_eligible_backlog: 12,
    replay_pending_count: 7,
    candidate_model_version: 9,
    champion_training_version: 5,
    candidate_replay_passes: 1,
  },
  learning: {},
  feed_metrics: {},
  agent_health: {
    status: "healthy",
    lag_seconds: 0,
    candidate: { status: "healthy", lag_seconds: 2, pending: 1 },
  },
  markets: [
    { key: "korea", label: "KRX", count: 1, fresh_count: 1 },
    { key: "us", label: "NASDAQ", count: 1, fresh_count: 1 },
  ],
  instruments: [
    {
      symbol: "005930.KS",
      name: "Samsung",
      group: "korea",
      fresh: true,
      quote: { close: 71000, date: new Date(dateNow).toISOString() },
      decision: { action: "HOLD", p_hold: 0.7 },
    },
    {
      symbol: "AAPL",
      name: "Apple",
      group: "us",
      fresh: false,
      quote: { close: 200 },
      decision: { action: "BUY", p_buy: 0.8 },
    },
  ],
};
const statusArg = process.argv.indexOf("--status-file");
const live =
  statusArg < 0
    ? fixture
    : JSON.parse(
        fs
          .readFileSync(process.argv[statusArg + 1], "utf8")
          .replace(/^\uFEFF/, ""),
      );
function build(isNew, data) {
  const dom = new JSDOM(
    (isNew ? newHtml : oldHtml).replace(/<script[^>]*>.*?<\/script>/gs, ""),
    { runScripts: "outside-only", url: "http://127.0.0.1:8766/" },
  );
  const w = dom.window,
    calls = [];
  const NativeDate = w.Date;
  w.Date = class extends NativeDate {
    constructor(...args) {
      super(...(args.length ? args : [dateNow]));
    }
    static now() {
      return dateNow;
    }
  };
  w.setInterval = () => 0;
  w.requestAnimationFrame = (callback) => callback();
  w.scrollTo = () => {};
  w.confirm = () => false;
  w.fetch = async (url, options = {}) => {
    if (options.method === "POST") {
      const payload = JSON.parse(options.body);
      calls.push({ url, payload });
      if (url === "/api/modes") Object.assign(data, payload);
    }
    return {
      ok: true,
      json: async () => (url === "/api/status" ? data : { ok: true }),
    };
  };
  // Initialization is tested in the real browser; unit cases drive exact render states.
  for (const code of isNew ? newCode : oldCode)
    vm.runInContext(
      code.replace(
        'document.addEventListener("DOMContentLoaded", startDashboard, { once: true });',
        "",
      ),
      dom.getInternalVMContext(),
    );
  return { dom, w, calls };
}
function canonical(el) {
  const clone = el.cloneNode(true),
    walker = el.ownerDocument.createTreeWalker(clone, 4),
    nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const n of nodes) {
    if (!n.textContent.trim()) n.remove();
    else if (!n.parentElement?.closest("pre") && el.id !== "runtimeUpdatesBody")
      n.textContent = n.textContent.replace(/\s+/g, " ").trim();
  }
  return clone.innerHTML;
}
function fields(doc) {
  const result = {};
  for (const el of doc.querySelectorAll("[id]")) {
    if (
      el.querySelector("[id]") ||
      (el.closest("details") && !el.closest("details").open)
    )
      continue;
    result[el.id] = [
      canonical(el),
      el.className,
      el.disabled,
      el.value,
      el.title,
      el.getAttribute("aria-pressed"),
    ];
  }
  return result;
}
const scenarios = {
  live,
  stopped: {
    ...live,
    running: false,
    agent_running: false,
    agent_process_running: false,
    feed_running: false,
  },
  paused: {
    ...live,
    observe_enabled: false,
    paper_enabled: false,
    learning_enabled: false,
  },
  error: {
    ...live,
    candidate_live_account: {
      ...live.candidate_live_account,
      status: "error",
      error: "fixture error",
    },
    metrics: { ...live.metrics, last_candidate_error: "fixture error" },
  },
  missing: {
    metrics: {},
    learning: {},
    agent_health: {},
    running: false,
    agent_process_running: false,
  },
};
function checkStaticContract() {
  const oldDoc = new JSDOM(oldHtml).window.document,
    newDoc = new JSDOM(newHtml).window.document;
  const ids = (doc) =>
    [...doc.querySelectorAll("[id]")].map((el) => el.id).sort();
  assert(
    ids(oldDoc).every((id) => ids(newDoc).includes(id)),
    "All existing HTML IDs are preserved",
  );
  assert.equal(newDoc.querySelectorAll("[data-view]").length, 7);
  const routes = (code) =>
    [
      ...new Set(
        [...code.join("\n").matchAll(/["'](\/api\/[^"']+)["']/g)].map(
          (match) => match[1],
        ),
      ),
    ].sort();
  assert(routes(oldCode).every(route => routes(newCode).includes(route)), "All existing API routes are preserved");
  assert.equal(newDoc.querySelectorAll("button[id]").length, 13);
  assert(newDoc.querySelectorAll("details").length >= 22);
  assert(
    !/\brender\w*\s*=\s*function/.test(newCode.join("\n")),
    "Render functions must not be redefined",
  );
  return {
    buttons: 13,
    expandablePanels: newDoc.querySelectorAll("details").length + 1,
    apiRoutes: routes(newCode).length,
  };
}
async function main() {
  const contract = checkStaticContract();
  let fieldChecks = 0;
  const measurements = [];
  for (const [name, data] of Object.entries(scenarios))
    for (const open of [false, true]) {
      const before = build(false, structuredClone(data)),
        after = build(true, structuredClone(data));
      for (const obj of [before, after]) {
        obj.w.render(data);
        if (open) {
          obj.w.document
            .querySelectorAll("details")
            .forEach((el) => (el.open = true));
          obj.w.render(data);
        }
      }
      const a = fields(before.w.document),
        b = fields(after.w.document);
      if (open)
        assert(
          Object.keys(a)
            .filter(
              (id) =>
                ![
                  "championAccountOverview",
                  "candidateAccountOverview",
                ].includes(id),
            )
            .every((id) => id in b),
          `${name}: every original field remains available when expanded`,
        );
      for (const id of Object.keys(a)) {
        if (!(id in b)) continue;
        if (["observeBtn", "paperBtn", "learningBtn"].includes(id)) {
          assert.match(
            b[id][0],
            /판단 중|판단 중지|체결 허용|체결 중지|학습 허용|학습 중지|확인 중/,
          );
          const field = {
            observeBtn: "observe_enabled",
            paperBtn: "paper_enabled",
            learningBtn: "learning_enabled",
          }[id];
          if (typeof data[field] === "boolean")
            assert.deepEqual(
              b[id].slice(1),
              a[id].slice(1),
              name + ": independent toggle state",
            );
          else {
            assert.equal(b[id][2], true);
            assert.equal(b[id][5], "false");
          }
          fieldChecks++;
          continue;
        }
        if (["resetAccountsBtn", "serverRestartBtn"].includes(id)) {
          assert.match(
            b[id][0],
            id === "resetAccountsBtn"
              ? /두 장기 운영계좌 초기화/
              : /웹서버만 재시작/,
          );
          assert.deepEqual(b[id].slice(1, 4), a[id].slice(1, 4));
          fieldChecks++;
          continue;
        }
        if (
          ["candidateBadge", "dualLearningStatus", "gateStory"].includes(id)
        ) {
          assert(b[id][0].length > 0);
          fieldChecks++;
          continue;
        }
        if (["workflowInferenceState", "workflowPaperState"].includes(id)) {
          const expected =
            after.w.operatorMetrics(data)[
              id === "workflowInferenceState" ? "Inference" : "Paper"
            ].status;
          assert.equal(b[id][0], expected);
          assert.deepEqual(b[id].slice(1), a[id].slice(1));
          fieldChecks++;
          continue;
        }
        if (
          ["accountInputScope", "accountRewardHorizon"].includes(id) &&
          !data.account_observability
        ) {
          assert.match(b[id][0], /측정 없음/);
          fieldChecks++;
          continue;
        }
        if (["learningFlowHealth", "learningState"].includes(id)) {
          assert(b[id][0].length > 0, `${name}: meaningful learning status`);
          fieldChecks++;
          continue;
        }
        if (id === "learningFlowHealth" && a[id][0] !== b[id][0]) {
          // The only intended wording fix separates the two previously merged lag values.
          const expected =
            data.agent_health?.candidate?.lag_seconds == null
              ? !data.agent_process_running
                ? "정지"
                : "미측정"
              : Number(data.agent_health.candidate.lag_seconds).toFixed(0) +
                "초";
          const value = b[id][0];
          assert(
            value.includes(" · Candidate 지연 " + expected) ||
              value.includes("판단 OFF"),
          );
          assert.deepEqual(a[id].slice(1), b[id].slice(1));
        } else
          assert.deepEqual(
            b[id],
            a[id],
            `${name}/${open ? "expanded" : "collapsed"}: ${id}`,
          );
        fieldChecks++;
      }
      if (name === "live") {
        const run = (obj) => {
          const obs = new obj.w.MutationObserver(() => {});
          obs.observe(obj.w.document.body, {
            subtree: true,
            childList: true,
            attributes: true,
            characterData: true,
          });
          const times = [];
          let mutations = 0;
          for (let i = 0; i < 5; i++) {
            const start = performance.now();
            obj.w.render(data);
            times.push(performance.now() - start);
            mutations += obs.takeRecords().length;
          }
          obs.disconnect();
          times.sort((x, y) => x - y);
          return {
            mutationsPerRefresh: mutations / 5,
            medianMs: +times[2].toFixed(2),
          };
        };
        measurements.push({
          details: open ? "expanded" : "collapsed",
          before: run(before),
          after: run(after),
        });
      }
      before.w.close();
      after.w.close();
    }
  // Modes must send only their own flag for all eight initial mode combinations.
  let actions = 0;
  for (let mask = 0; mask < 8; mask++)
    for (const [id, key] of [
      ["observeBtn", "observe_enabled"],
      ["paperBtn", "paper_enabled"],
      ["learningBtn", "learning_enabled"],
    ]) {
      const data = {
        ...fixture,
        observe_enabled: !!(mask & 1),
        paper_enabled: !!(mask & 2),
        learning_enabled: !!(mask & 4),
      };
      const obj = build(true, data);
      obj.w.render(data);
      const expected = !data[key];
      await obj.w.document.getElementById(id).onclick();
      assert.deepEqual(obj.calls, [
        { url: "/api/modes", payload: { [key]: expected } },
      ]);
      actions++;
      obj.w.close();
    }
  // Search and receive-only filters keep the same rows and counts.
  for (const isNew of [false, true]) {
    const obj = build(isNew, structuredClone(fixture));
    obj.w.render(fixture);
    const search = obj.w.document.getElementById("symbolSearch");
    search.value = "Apple";
    search.oninput();
    assert.match(
      obj.w.document.getElementById("decisionRows").textContent,
      /Apple/,
    );
    assert(
      !obj.w.document
        .getElementById("decisionRows")
        .textContent.includes("Samsung"),
    );
    search.value = "";
    search.oninput();
    obj.w.document.getElementById("freshOnlyBtn").onclick();
    assert(
      !obj.w.document
        .getElementById("decisionRows")
        .textContent.includes("Apple"),
    );
    obj.w.close();
  }
  // New view routing and score cards do not submit control commands.
  const routed = build(true, structuredClone(fixture));
  const sameConditions = Object.fromEntries(
    [
      "same_starting_cash",
      "same_market_input",
      "same_market_timeline",
      "same_last_bar",
      "same_cost_rules",
      "same_action_rule",
    ].map((key) => [key, true]),
  );
  const trialData = {
    ...fixture,
    validation_comparison: {
      ...sameConditions,
      active: true,
      accounts_available: true,
      bars_current: 39,
      bars_required: 390,
      champion: {
        KRW: { net_return_rate: 0.01 },
        USD: { net_return_rate: 0.03 },
      },
      candidate: {
        KRW: { net_return_rate: 0.02 },
        USD: { net_return_rate: 0.04 },
      },
    },
  };
  routed.w.render(trialData);
  assert.equal(
    routed.w.document.getElementById("trialChampionScoreNow").textContent,
    "2.0000%",
  );
  assert.equal(
    routed.w.document.getElementById("trialCandidateScoreNow").textContent,
    "3.0000%",
  );
  assert.match(
    routed.w.document.getElementById("trialLeader").textContent,
    /Candidate.*잠정/,
  );
  assert.equal(
    routed.w.document.getElementById("trialProgressMeter").value,
    39,
  );
  for (const [hash, page] of Object.entries({
    control: "overview",
    markets: "market",
    learning: "learning",
    promotionTrial: "trial",
    connection: "connection",
    system: "details",
  })) {
    routed.w.history.replaceState(null, "", "/#" + hash);
    routed.w.selectDashboardPage();
    const visible = [
      ...routed.w.document.querySelectorAll("[data-view]"),
    ].filter((el) => !el.hidden);
    assert.equal(visible.length, 1);
    assert.equal(visible[0].dataset.view, page);
  }
  assert.deepEqual(routed.calls, []);
  routed.w.close();
  console.log(
    JSON.stringify(
      {
        contract,
        renderCases: 10,
        fieldChecks,
        independentModeActions: actions,
        searchAndFreshFilters: "passed",
        viewRoutingAndTrialScores: "passed",
        measurements,
      },
      null,
      2,
    ),
  );
}
module.exports = { build, fixture };
if (require.main === module)
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
