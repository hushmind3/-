"use strict";

// Recipe presentation only; API, DOM diffing and formatting stay in app.js.
let assemblyState = null;
let assemblyRequest = null;
let assemblyCommandPending = false;
const assemblyStages = {queued:"대기열",ready:"시험 대기",replay:"replay 예선",paper:"paper 비교",qualified:"비교 통과 · 승격 대기",promoted:"승격",rejected:"탈락",blocked:"시험 보류",paused:"시험 중지",champion:"Champion"};

function renderAssembly(data) {
  assemblyState=data;
  badge("systemBadge","자동실험 "+(data.enabled ? "실행 중":"정지"),data.enabled ? "good":"");
  const candidate=data.candidate || {}, champion=data.champion || {};
  const board=[["자동화",data.enabled ? "실행 중":"정지"],["Champion",champion.candidate_id || "기준 준비 중"],
    ["Candidate",candidate.candidate_id || "없음"],["대기 후보",whole(data.queue.length)+"개"],
    ["누적 시험",whole(data.experiments)+"회"],["승격",whole(data.promotions)+"회"],
    ["탈락",whole(data.rejections)+"회"],["새 Expert",whole(data.new_experts.length)+"개"]];
  html("assemblyBoard",()=>board.map(([title,value])=>'<article class="status-card"><div class="status-top"><span class="status-title">'+esc(title)+'</span></div><div class="big assembly-number">'+esc(value)+'</div></article>').join(""));
  text("assemblyMessage",data.message || "");
  for (const button of document.querySelectorAll("[data-assembly-setting]")) {
    const key=button.dataset.assemblySetting, enabled=data.settings[key];
    const label={auto_replace:"자동 Candidate 교체",auto_promote:"자동 승격",detect_experts:"새 Expert 자동감지"}[key];
    button.textContent=label+": "+(enabled ? "ON · 끄기":"OFF · 켜기");
    button.setAttribute("aria-checked",String(enabled));
    button.classList.toggle("active",enabled);
  }
  for (const button of document.querySelectorAll("[data-assembly-action], [data-assembly-setting]")) button.disabled=assemblyCommandPending;
  const on=(recipe,id)=>recipe.enabled_experts?.includes(id) ? "사용":"OFF";
  html("assemblyExperts",()=>accountTable(["전문가","Champion","Candidate","역할","담당 universe","최근 선택","refresh"],data.experts.map(expert=>{
    const universe=expert.universe ? expert.universe.length>6 ? expert.universe.slice(0,3).join(", ")+" 외 "+(expert.universe.length-3)+"종목" : expert.universe.join(", ") : expert.description || "범용 시장 입력 · native 규격 유지";
    const recent=data.worker?.selected_experts?.includes(expert.id) ? "선택됨":"—";
    return [expert.name+(data.new_experts.includes(expert.id) ? " · NEW":""),on(champion,expert.id),on(candidate,expert.id),expert.role === "policy" ? "매매 정책":"시장 인식",universe,recent,
      "C "+(champion.refresh_seconds?.[expert.id] ?? "—")+" / 후보 "+(candidate.refresh_seconds?.[expert.id] ?? "—")+"초"];
  })));
  const rows=Object.entries(candidate.scores || {}).map(([phase,result])=>{
    const c=result.candidate || {},ch=result.champion || {};
    return [phase === "replay" ? "replay 예선":"paper 비교",accountPercent(c.net_return),accountPercent(ch.net_return),accountPercent(result.delta),whole(c.trades),accountMoney((c.fees || 0)+(c.slippage || 0),"USD"),accountPercent(c.max_drawdown),decimal(c.seconds,2)+"초"];
  });
  html("assemblyCandidate",()=>'<p><strong>'+esc(candidate.candidate_id || "후보 없음")+'</strong> · 부모 '+esc(candidate.parent_id || "—")+'</p><p>변경: '+esc(candidate.mutation_description || "—")+'</p><p>현재 단계: '+esc(assemblyStages[candidate.evaluation_state] || "준비 중")+'</p><p>'+esc(candidate.reason || data.worker?.message || "")+'</p>'+accountTable(["단계","후보 수익률","Champion","차이","체결","비용","최대 손실폭","실행시간"],rows.length ? rows:[["미측정","—","—","—","—","—","—","—"]])+'<p class="metric-caption">전체 PT 복제 '+whole(data.checkpoint_copies)+'개 · 작은 state '+decimal(data.candidate_state_bytes/1048576,2)+' MiB · 시험 중 탐험/학습 OFF</p>');
  html("assemblyQueue",()=>accountTable(["후보","부모","변경점","상태"],data.queue.map(recipe=>[recipe.candidate_id,recipe.parent_id,recipe.mutation_description,assemblyStages[recipe.evaluation_state] || recipe.evaluation_state])));
  html("assemblyHistory",()=>accountTable(["시각","후보","사건","변경점","paper 차이","이유"],data.history.map(row=>[timeOf(row.time),row.candidate_id || "Registry",({generated:"생성",installed:"장착",rejected:"탈락",promoted:"승격",registry_changed:"Expert 감지"})[row.event] || row.event,row.mutation || "—",accountPercent(row.scores?.paper?.delta),row.reason])));
}
async function refreshAssembly() {
  if (assemblyRequest) return assemblyRequest;
  assemblyRequest=(async()=>{
    try {renderAssembly(await api("/api/assembly/status"));}
    catch(error) {text("assemblyActionStatus","조회 실패: "+error.message);}
    finally {assemblyRequest=null;}
  })();
  return assemblyRequest;
}
async function commandAssembly(action,payload={}) {
  if (assemblyCommandPending)return;
  assemblyCommandPending=true;
  for (const button of document.querySelectorAll("[data-assembly-action], [data-assembly-setting]"))button.disabled=true;
  text("assemblyActionStatus","요청 적용 중…");
  try {
    const result=await api("/api/assembly/"+action,payload);
    if (!result.ok)throw new Error(result.error || "요청 실패");
    text("assemblyActionStatus",result.message || "적용 완료");
  } catch(error) {text("assemblyActionStatus","적용 실패: "+error.message);}
  finally {assemblyCommandPending=false;await refreshAssembly();}
}
$("assembly").addEventListener("click",event=>{
  const button=event.target.closest("button");
  if (!button)return;
  if (button.dataset.assemblyAction)commandAssembly(button.dataset.assemblyAction);
  else if (button.dataset.assemblySetting && assemblyState) {
    const key=button.dataset.assemblySetting;
    commandAssembly("settings",{[key]:!assemblyState.settings[key]});
  }
});
