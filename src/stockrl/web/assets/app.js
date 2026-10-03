"use strict";

// Application state, API, DOM change checks, polling and render composition.

const nodes = new Map();
// Feed liveness and model liveness are separate. All operational cards use this.
function modelIsRunning(data, role) {
  const runtime = data.model_runtime?.[role];
  return runtime
    ? runtime.loaded === true && runtime.status === "running"
    : data.agent_process_running === true;
}
function runningModelRoles(data) {
  return ["champion", "candidate"].filter((role) => modelIsRunning(data, role));
}
function modelIsLearning(data, role) {
  return modelIsRunning(data, role) && data.learning_enabled !== false
    && data.model_runtime?.[role]?.memory_scope !== "worker" && data.metrics?.[role + "_training"] === true;
}
function modelStateLabel(runtime) {
  return ({stopped: "정지", loading: "로딩 중", running: "실행 중", saving: "저장 중", error: "정지 · 오류"})[runtime?.status] || "상태 미확인";
}
const $ = (id) => {
  if (!nodes.has(id)) {
    const node = document.getElementById(id);
    if (node) nodes.set(id, node);
  }
  return nodes.get(id);
};

const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );

const num = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
};

const whole = (value) => Math.round(num(value)).toLocaleString("ko-KR");

const decimal = (value, digits = 3) =>
  value == null || value === ""
    ? "—"
    : Number.isFinite(Number(value))
      ? Number(value).toFixed(digits)
      : "—";

const SYMBOL_NAMES = {
  "005930.KS": "삼성전자",
  "000660.KS": "SK하이닉스",
  "035420.KS": "NAVER",
  "005380.KS": "현대차",
  "051910.KS": "LG화학",
  "005935.KS": "삼성전자우",
  "402340.KS": "SK스퀘어",
  "0010S0.KQ": "와이즈플래닛컴퍼니",
  "009150.KS": "삼성전기",
  "034020.KS": "두산에너빌리티",
  "240810.KQ": "원익IPS",
  "356680.KQ": "엑스게이트",
  "0161M0.KQ": "네오사피엔스",
  "006400.KS": "삼성SDI",
  "030530.KQ": "원익홀딩스",
  "353200.KS": "대덕전자",
  "036930.KQ": "주성엔지니어링",
  "047040.KS": "대우건설",
  "072950.KQ": "빛샘전자",
  "010170.KQ": "대한광통신",
  "096770.KS": "SK이노베이션",
  "000720.KS": "현대건설",
  "478340.KQ": "나라스페이스테크놀로지",
  "012450.KS": "한화에어로스페이스",
  "387690.KQ": "레메디",
  "003490.KS": "대한항공",
  "000150.KS": "두산",
  "105560.KS": "KB금융",
  "066570.KS": "LG전자",
  "009830.KS": "한화솔루션",
  "028050.KS": "삼성E&A",
  "058470.KQ": "리노공업",
  "007660.KS": "이수페타시스",
  "093370.KS": "후성",
  "003670.KS": "포스코퓨처엠",
  "440110.KQ": "파두",
  "028260.KS": "삼성물산",
  "055550.KS": "신한지주",
  "046970.KQ": "우리로",
  "001440.KS": "대한전선",
  "000270.KS": "기아",
  "001820.KS": "삼화콘덴서",
  "042700.KS": "한미반도체",
  "010060.KS": "OCI홀딩스",
  "222800.KQ": "심텍",
  "086790.KS": "하나금융지주",
  "329180.KS": "HD현대중공업",
  "006360.KS": "GS건설",
  "103590.KS": "일진전기",
  "386380.KQ": "스카이랩스",
  "115440.KQ": "우리넷",
  "950260.KQ": "인제니아테라퓨틱스(Reg.S)",
  "319660.KQ": "피에스케이",
  "298040.KS": "효성중공업",
  "373220.KS": "LG에너지솔루션",
  "034730.KS": "SK",
  "067310.KQ": "하나마이크론",
  "403870.KQ": "HPSP",
  "043260.KQ": "성호전자",
  "201490.KQ": "미투온",
  "396300.KQ": "HT로보틱스",
  "272210.KS": "한화시스템",
  "042660.KS": "한화오션",
  "080220.KQ": "제주반도체",
  "419050.KQ": "삼기에너지솔루션즈",
  "003230.KS": "삼양식품",
  "012330.KS": "현대모비스",
  "010120.KS": "LS ELECTRIC",
  "278470.KS": "에이피알",
  "128940.KS": "한미약품",
  "032830.KS": "삼성생명",
  "089970.KQ": "브이엠",
  "036540.KQ": "SFA반도체",
  "108490.KQ": "로보티즈",
  "052690.KS": "한전기술",
  "393210.KQ": "토마토시스템",
  "005490.KS": "POSCO홀딩스",
  "011070.KS": "LG이노텍",
  "010950.KS": "S-Oil",
  "095610.KQ": "테스",
  AAPL: "Apple",
  MSFT: "Microsoft",
  NVDA: "NVIDIA",
  AMZN: "Amazon",
  GOOGL: "Alphabet A",
  GOOG: "Alphabet C",
  CRM: "Salesforce",
  TSLA: "Tesla",
  AVGO: "Broadcom",
  COST: "Costco",
  NFLX: "Netflix",
  AMD: "AMD",
  INTC: "Intel",
  QCOM: "Qualcomm",
  CSCO: "Cisco",
  ADBE: "Adobe",
  TXN: "Texas Instruments",
  AMAT: "Applied Materials",
  MU: "Micron",
  KLAC: "KLA",
  LRCX: "Lam Research",
  ASML: "ASML",
  ADP: "ADP",
  BKNG: "Booking Holdings",
  PDD: "PDD Holdings",
  INTU: "Intuit",
  ISRG: "Intuitive Surgical",
  REGN: "Regeneron",
  VRTX: "Vertex Pharma",
  GILD: "Gilead",
  AMGN: "Amgen",
  SBUX: "Starbucks",
  PYPL: "PayPal",
  DIS: "Disney",
  PANW: "Palo Alto Networks",
  HD: "Home Depot",
  MDB: "MongoDB",
  JNJ: "Johnson & Johnson",
  KO: "Coca-Cola",
  PEP: "PepsiCo",
  MAR: "Marriott",
  MCD: "McDonald's",
  MELI: "MercadoLibre",
  LULU: "Lululemon",
  CHTR: "Charter Communications",
  TEAM: "Atlassian",
  PCAR: "PACCAR",
  ORLY: "O'Reilly Automotive",
  ROST: "Ross Stores",
  KDP: "Keurig Dr Pepper",
  NXPI: "NXP Semiconductors",
  FANG: "Diamondback Energy",
  MCHP: "Microchip Technology",
  FAST: "Fastenal",
  HON: "Honeywell",
  IBM: "IBM",
  JPM: "JPMorgan Chase",
  BAC: "Bank of America",
  WMT: "Walmart",
  XOM: "Exxon Mobil",
  CVX: "Chevron",
  UNH: "UnitedHealth",
  CAT: "Caterpillar",
  GS: "Goldman Sachs",
  SPY: "S&P 500 ETF",
  QQQ: "Nasdaq 100 ETF",
  "^GSPC": "S&P 500 Index",
  "^IXIC": "NASDAQ Composite",
  "^DJI": "Dow Jones Index",
  "^N225": "Nikkei 225",
  "^HSI": "Hang Seng Index",
  "^GDAXI": "DAX Index",
  "^FTSE": "FTSE 100",
  "ES=F": "S&P 500 Futures",
  "NQ=F": "NASDAQ 100 Futures",
  "GC=F": "Gold Futures",
  "SI=F": "Silver Futures",
  "CL=F": "WTI Crude Futures",
  "HG=F": "Copper Futures",
  "EURUSD=X": "EUR/USD",
  "KRW=X": "USD/KRW",
  "JPY=X": "USD/JPY",
  "GBPUSD=X": "GBP/USD",
  "^TNX": "US 10Y Yield",
  "^FVX": "US 5Y Yield",
  "^IRX": "US 3M Yield",
  "^TYX": "US 30Y Yield",
  "^VIX": "VIX Volatility Index",
  "BTC-USD": "Bitcoin/USD",
  "ETH-USD": "Ethereum/USD",
};

let current = null,
  selectedMarket = "all",
  freshOnly = false,
  environmentTouched = false;

function dateOf(value) {
  if (!value) return null;
  let raw = String(value).replace(/(\.\d{3})\d+/, "$1");
  if (!/[zZ]|[+-]\d\d:\d\d$/.test(raw)) raw += "Z";
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? null : date;
}

const clockFormat = new Intl.DateTimeFormat("ko-KR", {
  timeZone: "Asia/Seoul",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
function timeOf(value) {
  const date = dateOf(value);
  return date ? clockFormat.format(date) : "기록 없음";
}

function ageOf(value) {
  const date = dateOf(value);
  if (!date) return "수신 기록 없음";
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  return seconds < 60
    ? seconds + "초 전"
    : seconds < 3600
      ? Math.floor(seconds / 60) + "분 전"
      : Math.floor(seconds / 3600) + "시간 전";
}

function recent(value, seconds = 300) {
  const date = dateOf(value);
  return (
    !!date &&
    Date.now() - date.getTime() < seconds * 1000 &&
    Date.now() >= date.getTime() - 60000
  );
}

// A render frame commits each final text value once. Repeated summaries never
// flash intermediate values or rewrite an unchanged node.
let pendingText = null;
let pendingProperties = null;
const htmlValues = new WeakMap();
const detailOwners = new WeakMap();
const pageNodes = new Map();
function screenVisible(name) {
  if (!pageNodes.has(name))
    pageNodes.set(name, document.querySelector(`[data-view="${name}"]`));
  const node = pageNodes.get(name);
  return Boolean(node && !node.hidden);
}
function beginView() {
  pendingText = new Map();
  pendingProperties = new Map();
}
function detailsOpen(id) {
  const node = $(id);
  if (!detailOwners.has(node)) {
    const parents = [];
    for (let parent = node.parentElement; parent; parent = parent.parentElement)
      if (parent.tagName === "DETAILS" || parent.hasAttribute("data-view")) {
        // A closed panel still exposes its summary; outer closed panels stay lazy.
        if (
          parent.tagName === "DETAILS" &&
          node.closest("summary")?.parentElement === parent
        )
          continue;
        parents.push(parent);
      }
    detailOwners.set(node, parents);
  }
  return detailOwners
    .get(node)
    .every((parent) =>
      parent.tagName === "DETAILS" ? parent.open : !parent.hidden,
    );
}
function property(id, key, value) {
  if (pendingProperties) {
    if (!pendingProperties.has(id)) pendingProperties.set(id, new Map());
    pendingProperties.get(id).set(key, value);
  } else if ($(id)[key] !== value) $(id)[key] = value;
}
function attribute(id, key, value) {
  if ($(id).getAttribute(key) !== value) $(id).setAttribute(key, value);
}
function toggleClass(id, name, on) {
  if ($(id).classList.contains(name) !== on) $(id).classList.toggle(name, on);
}
function styleWidth(id, width) {
  if ($(id).style.width !== width) $(id).style.width = width;
}
function textOf(id) {
  return pendingText?.get(id) ?? $(id).textContent;
}
function commitView() {
  const values = pendingText,
    properties = pendingProperties;
  pendingText = null;
  pendingProperties = null;
  for (const [id, fields] of properties)
    for (const [key, value] of fields) property(id, key, value);
  for (const [id, value] of values) {
    if (detailsOpen(id)) writeText(id, value);
  }
}
function writeText(id, value) {
  const node = $(id);
  if (node.textContent !== value) {
    node.textContent = value;
    htmlValues.delete(node);
  }
}
function text(id, value) {
  const next = String(value ?? "");
  if (pendingText) pendingText.set(id, next);
  else writeText(id, next);
}
function html(id, value) {
  pendingText?.delete(id);
  if (!detailsOpen(id)) return;
  const node = $(id),
    next = String((typeof value === "function" ? value() : value) ?? "");
  if (htmlValues.get(node) !== next) {
    node.innerHTML = next;
    htmlValues.set(node, next);
  }
}
function badge(id, value, tone = "") {
  property(id, "className", "pill " + tone);
  text(id, value);
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw Error("서버 응답을 읽지 못했습니다.");
  }
  if (!response.ok) throw Error(data.error || data.message || "요청 실패");
  return data;
}

function setClock() {
  text("clock", clockFormat.format(new Date()) + " KST");
}

let lastDecisionRenderKey = null;

let statusRequest = null;
function refresh() {
  if (!statusRequest)
    statusRequest = readStatus().finally(() => {
      statusRequest = null;
    });
  return statusRequest;
}
async function readStatus() {
  if (screenVisible("trading-moe")) return refreshTradingMoE();
  if (screenVisible("experts")) return refreshExperts();
  try {
    render(await api("/api/status"));
    return true;
  } catch (error) {
    badge("systemBadge", "서버 연결 끊김", "bad");
    renderLearningSituation({ ...current, status_unavailable: true });
    renderWorkflowStatus({ status_unavailable: true });
    renderProcessHealth({ status_unavailable: true });
    text("runTitle", "서버 연결 실패");
    text("runDetail", error.message);
    feedback("로컬 서버 상태를 읽지 못했습니다. 5초 후 다시 확인합니다.", true);
    return false;
  }
}

// Screen ownership is explicit. Hidden pages do no table/account work.
function render(d) {
  current = d;
  beginView();
  try {
    renderControls(d);
    if (screenVisible("overview")) {
      renderStatus(d);
      renderTelemetry(d);
      renderWorkflowStatus(d);
      renderProcessHealth(d);
      renderPaperResults(d);
      renderLiveAccounts(d);
      renderOperationsOverview(d);
      renderObservationStatus(d);
      renderOverviewExperience(d);
    }
    if (screenVisible("market")) {
      renderMarkets(d);
      renderDecisions(d);
      renderOutputDiagnostics(d);
      renderVenues(d);
    }
    if (screenVisible("learning")) {
      renderLearningTotals(d);
      renderLearningMeasurements(d);
      renderLearningMetrics(d);
      renderDailyLearning(d);
      renderDualLearning(d);
      renderExperienceFlow(d);
      renderLearnerSummary(d);
    }
    if (screenVisible("trial")) {
      renderValidationAccounts(d);
      renderPromotionHistory(d);
      renderTrialSummary(d);
    }
    if (screenVisible("connection")) renderConnection(d);
    if (screenVisible("details")) {
      renderTelemetry(d);
      renderGpuScheduler(d);
      renderAccountInputContext(d);
      renderProcessingHistory(d);
      renderInferenceWork(d);
      renderRuntimeUpdates(d);
      renderDailyOperation(d);
      if (detailsOpen("logs")) text("logs", d.logs || "운영 기록 없음");
    }
    if (
      screenVisible("learning") ||
      screenVisible("details") ||
      detailsOpen("inputQuickSummary")
    )
      renderDailyOperation(d);
    text(
      "lastData",
      "최근 시장 데이터 · " + timeOf(d.metrics?.last_market_timestamp),
    );
  } finally {
    commitView();
  }
}

// Start after the screen scripts have loaded. UI updates never restart the models.
function startDashboard() {
  // Old tabs can receive newer assets during deployment. Reload the document
  // before running renderers that require the corresponding screen markup.
  if (document.documentElement.dataset.uiRevision !== "cards-layout-2") {
    const url = new URL(location.href);
    if (url.searchParams.get("ui") !== "cards-layout-2") {
      url.searchParams.set("ui", "cards-layout-2");
      location.replace(url.href);
    } else {
      writeText("systemBadge", "화면 버전 불일치 · 새로고침 필요");
    }
    return;
  }
  selectDashboardPage();
  window.addEventListener("hashchange", selectDashboardPage);
  setClock();
  setInterval(setClock, 1000);
  refresh();
  setInterval(() => {
    if (!document.hidden) refresh();
  }, 5000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refresh();
  });
  document.addEventListener(
    "toggle",
    (event) => {
      if (event.target.tagName === "DETAILS" && event.target.open && current)
        render(current);
    },
    true,
  );
}

// Navigation changes visibility only; all controls retain their state and API.
function selectDashboardPage() {
  const target = location.hash.slice(1) || "control";
  const element = document.getElementById(target);
  const page = element?.closest("[data-view]")?.dataset.view || "overview";
  for (const view of document.querySelectorAll("[data-view]"))
    view.hidden = view.dataset.view !== page;
  for (const link of document.querySelectorAll(".sidebar a[data-page]")) {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (current) render(current);
  if (page === "experts") refreshExperts();
  if (page === "trading-moe") refreshTradingMoE();
  requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "instant" }));
}
document.addEventListener("DOMContentLoaded", startDashboard, { once: true });

// Shared presentation only: these helpers own no account or trial state.
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
