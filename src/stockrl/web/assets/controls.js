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
  badge(
    "systemBadge",
    d.running
      ? d.agent_running && d.feed_running
        ? "시스템 실행 중"
        : "시스템 일부 대기"
      : "시스템 정지",
    d.running ? (d.agent_running && d.feed_running ? "good" : "warn") : "bad",
  );
  badge(
    "appliedBadge",
    changing ? "시스템 전환 중" : "여러 시간봉 통합 판단",
    changing ? "warn" : "good",
  );

  text(
    "applyBtn",
    d.restarting ? "적용 중…" : d.running ? "시스템 실행 중" : "시스템 시작",
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
  "agentReloadBtn",
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
    feedback("여러 시간봉을 통합하는 시장 관찰과 모델 판단을 시작합니다.");
    await api("/api/start", { mode: "live" });
    feedback("요청을 보냈습니다. 실행 상태를 확인하고 있습니다.");
  });

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

$("agentReloadBtn").onclick = () =>
  runCommand($("agentReloadBtn"), async () => {
    feedback(
      "모델이 현재 작업을 저장한 뒤 모델 프로세스만 다시 적용됩니다. 시세 수집과 웹은 유지됩니다.",
    );
    const result = await api("/api/agent/reload", {});
    feedback(
      result.pending
        ? "승급전이 끝나면 자동으로 재적용합니다."
        : "모델이 상태를 저장한 후 재적용됩니다.",
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
  const validationActive = !!d.metrics?.candidate_validation_active,
    reloadPending = !!d.agent_reload_pending;
  property(
    "agentReloadBtn",
    "disabled",
    !!pendingCommand ||
      changing ||
      !d.running ||
      !d.agent_process_running ||
      reloadPending,
  );
  property(
    "agentReloadBtn",
    "title",
    reloadPending
      ? "재적용은 승급전 종료 후 자동 적용됩니다."
      : validationActive
        ? "승급전 중 재적용을 예약합니다."
        : "저장 후 모델만 재시작합니다. 시세 수집과 웹은 유지됩니다.",
  );
  for (const mode of modeDefinitions) {
    const known = typeof d[mode.field] === "boolean",
      on = d[mode.field];
    const label = known
      ? mode.label + ": " + (on ? mode.on + " · 중지" : mode.off + " · 시작")
      : mode.label + ": 확인 중";
    text(
      mode.id,
      pendingCommand?.id === mode.id ? pendingCommand.label : label,
    );
    toggleClass(mode.id, "active", known && on);
    attribute(mode.id, "aria-pressed", String(known && on));
    property(mode.id, "disabled", !!pendingCommand || changing || !known);
  }
  text(
    "runTitle",
    d.running
      ? observe
        ? "모델 판단 실행 중"
        : "모델 판단 중지 · 시세 저장 계속"
      : "시스템 정지",
  );
  text(
    "runDetail",
    "판단 " +
      (observe ? "ON" : "OFF") +
      " | 체결 " +
      (paper ? "ON" : "OFF") +
      " | 학습 " +
      (learning ? "ON" : "OFF") +
      " | 실제 주문 OFF",
  );
  text(
    "autonomyDetail",
    "판단: 두 모델의 새 추론·승급전. 체결: 기존 주문 실행·새 판단 주문 접수. 학습: 저장된 replay로 두 모델 업데이트. 각각 독립 제어합니다. 판단 OFF에서도 시세 저장·기존 보유 평가·replay 학습은 계속됩니다. 비트코인·환율 등 문맥용 시세만 바뀌면 풀 추론하지 않습니다.",
  );
}

// The three flags are independent; the status response confirms each change.
const modeDefinitions = [
  {
    id: "observeBtn",
    field: "observe_enabled",
    label: "모델 판단",
    on: "판단 중",
    off: "판단 중지",
    start: "두 모델의 판단 시작이 확인됐습니다.",
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
    stop: "학습 중지 요청이 확인됐습니다. 진행 batch는 저장하고 미학습 경험은 보존됩니다.",
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
      mode.label + (enabled ? " 시작 요청 중" : " 중지 요청 중"),
    );
  };

function renderControls(d) {
  renderSystemControls(d);
  renderModeControls(d);
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
