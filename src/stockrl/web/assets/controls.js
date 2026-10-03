"use strict";

// All operator actions and independent mode controls. API contracts stay here.

function feedback(message, error = false) {
  const el = $("commandFeedback");
  el.hidden = false;
  el.className = "command-feedback" + (error ? " error" : "");
  el.textContent = message;
}

function connectResult(message, error = false) {
  const el = $("connectResult");
  el.className = "result show" + (error ? " error" : "");
  el.textContent = message;
}

function renderSystemControls(d) {
  const changing = !!(d.stopping || d.restarting);
  const roles = runningModelRoles(d);
  badge("systemBadge", d.status_unavailable ? "상태 미확인" : roles.length ? "모델 " + roles.length + "개 실행 중" : d.feed_running ? "시세 수집 중 · 모델 정지" : "시스템 정지", d.status_unavailable ? "bad" : roles.length || d.feed_running ? "good" : "");
  badge(
    "appliedBadge",
    changing ? "시스템 전환 중" : "시세 수집 · 모델 각각 시작",
    changing ? "warn" : "good",
  );

  text(
    "applyBtn",
    d.restarting ? "적용 중…" : d.feed_running ? "시세 수집 실행 중" : "시스템 시작 · 시세 수집",
  );
  property("applyBtn", "disabled", !!pendingCommand || changing || !!d.running);
  property(
    "reconnectBtn",
    "disabled",
    !!pendingCommand || changing || !d.running,
  );
  property("stopBtn", "disabled", !!pendingCommand || changing || !d.running);
}

function renderConnection(d) {
  const p = d.provider || {},
    f = d.feed_metrics || {},
    auth = p.last_test || {},
    connected = !!f.broker_connected,
    last = f.broker_last_message_utc;
  if (!environmentTouched)
    property("environment", "value", p.environment || "real");
  text("providerSaved", p.saved ? "키움 키 저장됨" : "저장된 키 없음");
  text(
    "connectionTitle",
    !p.saved
      ? "키 등록 필요"
      : !auth.ok
        ? "인증 확인 필요"
        : !connected
          ? "소켓 연결 대기"
          : !recent(last)
            ? "연결 완료 · 체결 대기"
            : "실시간 체결 수신 중",
  );
  text(
    "connectionDetail",
    !p.saved
      ? "키움 API 키를 입력해 연결하세요."
      : f.broker_error ||
          auth.message ||
          "인증과 시세 연결 상태를 아래에서 확인하세요.",
  );
  text(
    "authStatus",
    auth.ok
      ? "성공 · " + timeOf(auth.time)
      : auth.ok === false
        ? "실패 · " + (auth.message || "")
        : "미확인",
  );
  text("socketStatus", connected ? "연결됨" : "연결 안 됨");
  text(
    "tickStatus",
    recent(last)
      ? "수신 중 · " + ageOf(last)
      : last
        ? "마지막 " + timeOf(last)
        : "아직 체결 없음",
  );
  text("brokerSymbols", whole(f.broker_symbols) + "개");
  text(
    "providerEnvironment",
    p.environment === "real" ? "실전 시세" : "모의 시세",
  );
  property("recheckBtn", "disabled", !!pendingCommand || !p.saved);
}

const commandButtons = [
  "applyBtn",
  "reconnectBtn",
  "serverRestartBtn",
  "resetAccountsBtn",
  "stopBtn",
  "observeBtn",
  "paperBtn",
  "learningBtn",
  "connectBtn",
  "recheckBtn",
  "ipBtn",
];
let pendingCommand = null;
async function runCommand(button, task, requestLabel = "요청 중…") {
  if (!!pendingCommand) return;
  pendingCommand = {
    id: button.id,
    label: requestLabel,
    previous: button.textContent,
  };
  for (const id of commandButtons) property(id, "disabled", true);
  button.setAttribute("aria-busy", "true");
  text(button.id, requestLabel);
  feedback(requestLabel);
  try {
    const result = await task();
    if (result === false) return;
    if (statusRequest) await statusRequest;
    const confirmed = await refresh();
    if (!confirmed)
      throw Error(
        "요청은 전달됐지만 현재 상태를 확인하지 못했습니다. 재연결 후 적용 상태를 확인하세요.",
      );
    if (result?.mode) {
      if (current[result.mode] !== result.enabled)
        throw Error(
          "설정 응답과 현재 상태가 다릅니다. 다음 갱신에서도 적용 여부를 확인하세요.",
        );
      feedback(result.message);
    }
  } catch (error) {
    feedback(error.message, true);
  } finally {
    text(button.id, pendingCommand.previous);
    pendingCommand = null;
    button.removeAttribute("aria-busy");
    for (const id of commandButtons) property(id, "disabled", false);
    if (current) {
      renderControls(current);
      renderConnection(current);
    }
  }
}

$("symbolSearch").oninput = () => {
  if (current) renderDecisions(current);
};

$("freshOnlyBtn").onclick = () => {
  freshOnly = !freshOnly;
  if (current) renderDecisions(current);
};

$("environment").onchange = () => {
  environmentTouched = true;
};

$("applyBtn").onclick = () =>
  runCommand($("applyBtn"), async () => {
    feedback("시장 feed를 시작합니다. Champion과 Candidate는 각각 시작 버튼으로 실행합니다.");
    await api("/api/start", { mode: "live" });
    feedback("요청을 보냈습니다. 실행 상태를 확인하고 있습니다.");
  });

const modelCommandPending = new Set();
const modelCommandErrors = new Map();
for (const role of ["champion", "candidate"]) {
  for (const action of ["start", "stop"]) {
    $(role + "Model" + (action === "start" ? "Start" : "Stop")).onclick = async () => {
      if (modelCommandPending.has(role)) return;
      modelCommandPending.add(role);
      modelCommandErrors.delete(role);
      $(role + "ModelStart").disabled = $(role + "ModelStop").disabled = true;
      text(role + "ModelMessage", action === "start" ? "적재 요청 중" : "저장·해제 요청 중");
      try {
        const result = await api("/api/models/" + role + "/" + action, {});
        if (!result.ok) throw new Error(result.error || "요청 실패");
        text(role + "ModelMessage", action === "start" ? "적재 요청됨 · 상태를 확인합니다" : "현재 작업 후 저장하고 이 모델만 해제합니다");
      } catch (error) {
        modelCommandErrors.set(role, "요청 실패: " + error.message);
        text(role + "ModelMessage", "요청 실패: " + error.message);
      } finally {
        modelCommandPending.delete(role);
        await refresh();
      }
    };
  }
}

$("serverRestartBtn").onclick = () =>
  runCommand($("serverRestartBtn"), async () => {
    feedback("웹서버만 재시작합니다. 시세 수집과 모델 판단·학습은 계속됩니다.");
    await api("/api/server/restart", {});
    for (let i = 0; i < 30; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (r.ok) {
          window.location.reload();
          return;
        }
      } catch (error) {}
    }
    feedback(
      "웹서버 재연결을 확인하세요. 모델 프로세스는 별도로 확인할 수 있습니다.",
      true,
    );
  });

$("resetAccountsBtn").onclick = () =>
  runCommand($("resetAccountsBtn"), async () => {
    if (
      !window.confirm(
        "운영 Champion과 Candidate 관찰용 가상계좌의 보유 종목, 체결, 비용, 누적 손익을 초기화하고 시드머니로 되돌립니다. 모델 가중치와 replay는 유지됩니다. 현재 승급 검증이 진행 중이면 재시작 과정에서 해당 시험은 무효 처리됩니다. 계속할까요?",
      )
    )
      return false;
    feedback(
      "가상계좌를 초기화하고 실행 중인 시스템을 저장 후 다시 시작합니다.",
    );
    await api("/api/paper-accounts/reset", {});
    feedback("Champion과 Candidate 관찰 계좌를 시드머니로 초기화했습니다.");
  });

$("stopBtn").onclick = () =>
  runCommand($("stopBtn"), async () => {
    feedback("시스템을 정지하고 상태를 저장합니다.");
    await api("/api/stop", {});
    feedback("전체 정지가 적용됐습니다.");
  });

$("reconnectBtn").onclick = () =>
  runCommand($("reconnectBtn"), async () => {
    await api("/api/feed/reconnect", {});
    feedback("시세 수집기 재연결을 요청했습니다.");
  });

$("connectBtn").onclick = () =>
  runCommand($("connectBtn"), async () => {
    const app_key = $("appKey").value.trim(),
      secret = $("secret").value.trim();
    if (!app_key || !secret)
      throw Error("App Key와 App Secret을 함께 입력해 주세요.");
    connectResult("키움 인증 확인 중…");
    const result = await api("/api/provider/connect", {
      environment: $("environment").value,
      app_key,
      secret,
      account: $("account").value.trim(),
    });
    connectResult(
      result.message || (result.ok ? "인증 성공" : "인증 실패"),
      !result.ok,
    );
    if (result.ok) {
      property("appKey", "value", "");
      property("secret", "value", "");
      property("account", "value", "");
      environmentTouched = false;
    }
  });

$("recheckBtn").onclick = () =>
  runCommand($("recheckBtn"), async () => {
    connectResult("저장된 키를 검사하고 있습니다.");
    const result = await api("/api/provider/test", {
      provider: "kiwoom",
      environment: $("environment").value,
    });
    connectResult(result.message || "키 검사 완료", !result.ok);
    if (result.ok) environmentTouched = false;
  });

$("ipBtn").onclick = () =>
  runCommand($("ipBtn"), async () => {
    text("publicIp", "조회 중…");
    const result = await api("/api/provider/public-ip");
    text("publicIp", result.ip || "조회 실패");
  });

function renderModeControls(d) {
  if (detailsOpen("appliedSettings")) {
    const m = d.metrics || {},
      coverage = m.multiscale_coverage || {};
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
  }

  const changing = !!(d.stopping || d.restarting),
    observe = !!d.observe_enabled,
    paper = !!d.paper_enabled,
    learning = d.learning_enabled !== false;
  property("serverRestartBtn", "disabled", !!pendingCommand || changing);
  for (const mode of modeDefinitions) {
    const known = typeof d[mode.field] === "boolean",
      on = d[mode.field];
    const label = known
      ? mode.label + ": " + (on ? mode.on + " · 끄기" : mode.off + " · 켜기")
      : mode.label + ": 확인 중";
    text(
      mode.id,
      pendingCommand?.id === mode.id ? pendingCommand.label : label,
    );
    toggleClass(mode.id, "active", known && on);
    attribute(mode.id, "aria-pressed", String(known && on));
    property(mode.id, "disabled", !!pendingCommand || changing || !known);
  }
  const roles = runningModelRoles(d);
  text("runTitle", roles.length ? roles.map((role) => role === "champion" ? "Champion" : "Candidate").join(" · ") + " 실행 중" : d.feed_running ? "시세 수집 중 · 모델 정지" : "시스템 정지");
  text("runDetail", "실행 중인 모델에 적용 · 판단 " + (observe ? "허용" : "중지") + " | 가상 체결 " + (paper ? "허용" : "중지") + " | 경험 학습 " + (learning ? "허용" : "중지") + " | 실제 주문 OFF");
  text(
    "autonomyDetail",
    "모델 시작·정지와 아래 작업 허용은 별개입니다. 판단을 꺼도 시세 수집과 저장 경험 학습은 계속됩니다. 가상 체결을 끄면 새 주문과 대기 주문 체결을 중지하고 보유는 유지합니다. 학습을 끄면 가중치를 갱신하지 않고 경험을 보존합니다. TradingMoE 모델은 확보된 ETHUSDT 과거 구간을 사용하며, 전용 자동매매 화면은 별도로 제어합니다.",
  );
}

// The three flags are independent; the status response confirms each change.
const modeDefinitions = [
  {
    id: "observeBtn",
    field: "observe_enabled",
    label: "모델 판단",
    on: "판단 허용",
    off: "판단 중지",
    start: "새 판단을 허용했습니다. 모델은 Champion·Candidate 시작 버튼으로 각각 실행합니다.",
    stop: "새 판단·승급전 중지. 시세 저장·기존 보유 평가·학습 설정은 유지됩니다.",
  },
  {
    id: "paperBtn",
    field: "paper_enabled",
    label: "가상 체결",
    on: "체결 허용",
    off: "체결 중지",
    start: "가상 체결 허용이 확인됐습니다. 판단·학습 설정은 유지됩니다.",
    stop: "새 가상 체결 중지. 보유 평가·판단·학습 설정은 유지됩니다.",
  },
  {
    id: "learningBtn",
    field: "learning_enabled",
    label: "경험 학습",
    on: "학습 허용",
    off: "학습 중지",
    start: "두 모델의 replay 학습 허용이 확인됐습니다.",
    stop: "학습 중지 요청이 확인됐습니다. 진행 중인 학습 묶음은 저장하고 미학습 경험은 보존됩니다.",
  },
];
for (const mode of modeDefinitions)
  $(mode.id).onclick = () => {
    if (!current || current[mode.field] == null)
      return feedback("현재 설정을 먼저 확인해야 합니다.", true);
    const enabled = !current[mode.field];
    return runCommand(
      $(mode.id),
      async () => {
        await api("/api/modes", { [mode.field]: enabled });
        return {
          mode: mode.field,
          enabled,
          message: enabled ? mode.start : mode.stop,
        };
      },
      mode.label + (enabled ? " 허용 요청 중" : " 중지 요청 중"),
    );
  };

function renderModelControls(d) {
  for (const role of ["champion", "candidate"]) {
    const lifecycle = d.model_runtime?.[role];
    const active = modelIsRunning(d, role);
    if (!lifecycle) {
    text(role + "CommandState", "상태 확인 중");
    property(role + "ModelStart", "disabled", true);
    property(role + "ModelStop", "disabled", true);
      continue;
    }
    text(role + "CommandState", modelCommandPending.has(role) ? "요청 중" : modelStateLabel(lifecycle));
    property(role + "CommandState", "className", "pill " + (lifecycle.error ? "bad" : active ? "good" : lifecycle.status === "loading" || lifecycle.status === "saving" ? "warn" : ""));
    property(role + "ModelStart", "disabled", d.status_unavailable || !!pendingCommand || d.stopping || modelCommandPending.has(role) || lifecycle.requested && lifecycle.status !== "error" || lifecycle.status === "saving");
    property(role + "ModelStop", "disabled", d.status_unavailable || !!pendingCommand || d.stopping || modelCommandPending.has(role) || !lifecycle.requested && !lifecycle.loaded || lifecycle.status === "saving");
    if (!modelCommandPending.has(role)) text(role + "ModelMessage", modelCommandErrors.get(role) || lifecycle.error || (lifecycle.status === "stopped" ? "계좌 유지 · 모델 메모리 해제" : lifecycle.status === "saving" ? "현재 작업을 마친 뒤 저장·해제합니다" : lifecycle.status === "loading" ? "현재 파일을 적재하는 중입니다" : lifecycle.source || "선택한 판단·가상 체결·경험 학습을 실행합니다"));
  }
}

function renderControls(d) {
  renderSystemControls(d);
  renderModeControls(d);
  renderModelControls(d);
  if (pendingCommand) text(pendingCommand.id, pendingCommand.label);
}

// Delegation is installed once; market-card/filter HTML updates retain behavior.
$("marketGrid").onclick = (event) => {
  const card = event.target.closest("[data-market]");
  if (!card || !current) return;
  selectedMarket = card.dataset.market;
  renderDecisions(current);
  $("decisions").scrollIntoView({ behavior: "smooth" });
};
$("marketFilters").onclick = (event) => {
  const filter = event.target.closest("[data-filter]");
  if (!filter || !current) return;
  selectedMarket = filter.dataset.filter;
  renderDecisions(current);
};
