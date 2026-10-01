const renderBeforeAccountDiagnostics=render;
function renderDailyOperation(d){
 const input=d.input_availability||{},stored=input.stored||{},fresh=input.fresh_quotes||{},cycle=d.daily_cycle||{},v=d.validation_comparison||{},m=d.metrics||{};
 text("inputQuickSummary","등록 "+whole(input.configured)+" · 최근 수신 "+whole(input.fresh)+" · 모델 입력 "+(input.model_input==null?"확인 대기":whole(input.model_input))+" · 설정상 매매 가능 "+whole(input.configured_tradable)+" · 문맥 전용 "+whole(input.context_only)+" · 호가 "+whole(fresh.bid_ask_symbols)+" · 잔량 "+whole(fresh.book_size_symbols)+" · 시세/판단 지연 "+(d.agent_health?.lag_seconds==null?"미측정":whole(d.agent_health.lag_seconds)+"초"));
 text("dailyCycleSummary","하루 승급전 판정 → 다음 고정 시험 준비 · 다음 "+timeOf(cycle.next_reset_utc)+" KST · 장기 운용 계좌·모델·미학습 경험 유지"+(cycle.in_progress?" · 처리 중":"")+" · 마지막 수동 초기화 "+timeOf(cycle.last_reset_utc));
 const common=m.shared_observation||{},origins=common.experience_origins||{},hist=m.daily_history_input_status||{};
 const candidateHealth=d.agent_health?.candidate||{};
 text("sharedObservationSummary","새 구조의 공통 필수 관찰 "+whole(common.common)+"회 · Champion 완료 "+whole(common.common)+" · Candidate 완료 "+whole(common.candidate_completed)+" · Candidate 처리 대기 "+whole(common.pending)+" (삭제·건너뜀 없이 DB 보존) · Candidate 시세 지연 "+(candidateHealth.lag_seconds==null?"미측정":whole(candidateHealth.lag_seconds)+"초")+((candidateHealth.status==="stale"||candidateHealth.status==="error")?" · 확인 필요: "+candidateHealth.reason:"")+". 자체 경험 기록: Champion "+whole(origins.champion)+" / Candidate "+whole(origins.candidate)+". 같은 행동이어도 서로 다른 계좌 결과는 별도 경험입니다.");
 text("dailyHistorySummary","5년 이상 완료 일봉 확보 "+whole(hist.five_year_symbols)+" / "+whole(hist.observed_symbols)+"종목 · 일봉 없음 "+whole(hist.missing_symbols)+" · 인코더 최대 "+whole(hist.max_bars||1300)+"개 일봉. 최근 실제 장기 일봉 포함 학습: Champion "+whole(m.champion_last_completed_round?.daily_history_samples)+" / Candidate "+whole(m.candidate_last_completed_round?.daily_history_samples)+"개 경험. 부족분은 공급자 수집으로 보충 중이며 없는 과거를 학습했다고 세지 않습니다.");
 $("dailyHistoryCoverage").innerHTML=(hist.symbols||[]).length?accountTable(["종목","완료 일봉 수","입력 기간","5년 확보"],hist.symbols.map(row=>[instrumentLabel(row.symbol,d),whole(row.bars)+" / "+whole(hist.max_bars||1300),decimal(row.years,2)+"년",row.five_years_available?"확보":"부족 / 보충 또는 상장 기간 확인"])):"일봉 인코더 입력 측정 대기";
 const quoteRows=[["OHLCV","ohlcv_symbols"],["최우선 매수·매도 호가 / 스프레드","bid_ask_symbols"],["매수·매도 잔량 / 잔량 불균형","book_size_symbols"],["매수·매도 체결량 / 체결 불균형","directional_volume_symbols"],["체결 횟수","trade_count_symbols"]].map(([name,key])=>[name,whole(stored[key])+" / "+whole(input.configured),whole(fresh[key])+" / "+whole(input.configured)]);
 quoteRows.push(["1초 / 15초 / 30초 입력","미수집 / 미수집 / 미수집","미수집 / 미수집 / 미수집"]);
 $("microstructureAvailability").innerHTML=accountTable(["입력 항목","저장된 마지막 값","최근 5분 갱신"],quoteRows);
 const longRows=Object.entries(m.long_context_input_status||{}).map(([name,info])=>[name,accountPercent(info.mean_history_coverage),whole(info.available_symbols)+" / "+whole(info.observed_symbols),m.champion_last_completed_round?.long_context_samples?.[name]==null?"측정 대기":whole(m.champion_last_completed_round.long_context_samples[name]),m.candidate_last_completed_round?.long_context_samples?.[name]==null?"측정 대기":whole(m.candidate_last_completed_round.long_context_samples[name])]);
 $("longContextAvailability").innerHTML=longRows.length?accountTable(["문맥 길이","평균 확보율","가용 종목","Champion 최근 학습 경험","Candidate 최근 학습 경험"],longRows):"장기 문맥 측정 대기";
 const before=d.universe_expansion?.before||{},obs=d.account_observability||{},gpu=d.physical_gpu||{},latest=m.champion_last_completed_round||{},previous=before.champion_round||{};
 const timing=value=>value==null?"미측정":decimal(value,2)+"초",memory=value=>value==null?"미측정":decimal(value/1024**3,2)+"GiB";
 $("universeCapacityComparison").innerHTML=accountTable(["측정 항목","확대 직전 · "+timeOf(before.measured_utc),"현재"],[
 ["등록 / 최근 수신 / 모델 입력",whole(before.configured)+" / "+whole(before.fresh)+" / "+whole(before.model_input),whole(input.configured)+" / "+whole(input.fresh)+" / "+whole(input.model_input)],
 ["미학습 경험",whole(before.backlog),whole(m.replay_eligible_backlog)],
 ["시세와 판단 지연",timing(before.agent_health?.lag_seconds),timing(d.agent_health?.lag_seconds)],
 ["Champion 판단 시간",timing(before.champion_inference_seconds),timing(obs.champion?.last_inference_seconds)],
 ["Candidate 판단 시간",timing(before.candidate_inference_seconds),timing(obs.candidate?.last_inference_seconds)],
 ["Champion 최근 학습 경험 / optimizer 횟수",whole(previous.unique_samples)+" / "+whole(previous.optimizer_steps),whole(latest.unique_samples)+" / "+whole(latest.optimizer_steps)],
 ["Champion 실제 학습 계산 / 총시간",timing(previous.compute_seconds)+" / "+timing(previous.total_seconds),timing(latest.compute_seconds)+" / "+timing(latest.total_seconds)],
 ["학습 회차 최대 VRAM",memory(previous.peak_allocated_bytes),memory(latest.peak_allocated_bytes)],
 ["장치 GPU 사용률",before.physical_gpu?.utilization_percent==null?"미측정":whole(before.physical_gpu.utilization_percent)+"%",gpu.utilization_percent==null?"미측정":whole(gpu.utilization_percent)+"%"],
 ["최근 학습 회차 완료 시각",timeOf(previous.completed_utc),timeOf(latest.completed_utc)]]);
 const score=role=>Object.values(v[role]||{}).reduce((total,b)=>total+num(b.net_return_rate),0),cs=score("candidate"),bs=score("champion"),required=num(v.bars_required)||390,bars=num(v.bars_current);
 const health=v.same_market_input&&v.same_market_timeline&&v.same_starting_cash&&v.same_cost_rules&&v.same_action_rule;
 text("promotionDailyConditions","하루 승급전 · Champion 고정 v"+whole(v.champion_snapshot_version)+" / Candidate 고정 v"+whole(v.snapshot_version)+" · 시작 "+timeOf(v.started_utc||v.start_after)+" · 관측 "+whole(bars)+" / 최소 "+whole(required)+"개 시장 분. 통화별 시드 대비 수익률 합: Champion "+accountPercent(bs)+" / Candidate "+accountPercent(cs)+". 조건: Candidate 이익 "+(cs>0?"통과":"미충족")+" · Champion 초과 "+(cs>bs?"통과":"미충족")+" · 최소 관측 "+(bars>=required?"통과":"미충족")+" · 동일 입력/비용 "+(health?"확인":"대기/불일치")+". 중간 수익만으로 즉시 승급하지 않고 오전 "+whole(cycle.hour_kst??7)+"시에 판정합니다.");
 const reason=value=>({"candidate paper-account net return was not positive":"Candidate 비용 차감 이익 없음","candidate paper-account net return did not beat champion":"Champion 초과 성과 없음","sequential paper-account net return improved":"이익을 내고 Champion을 이겨 승급","sequential paper validation window is incomplete":"최소 시장 관측 부족","restart invalidated the in-memory validation snapshot":"재시작으로 RAM 시험본 무효"}[value]||value||"상세 사유 미기록");
 const history=(m.candidate_gate_history||[]).slice().reverse();
 $("promotionDecisionHistory").innerHTML=history.length?accountTable(["판정 시각","Champion / Candidate 버전","시장 관측","두 점수 · Champion / Candidate","결과","사유"],history.map(h=>[timeOf(h.time_utc),h.champion_version==null?"과거 미기록":"v"+whole(h.champion_version)+" / v"+whole(h.candidate_version),h.bars==null?"과거 미기록":whole(h.bars)+" / "+whole(h.required_bars),accountPercent(h.champion_score)+" / "+accountPercent(h.candidate_score),h.applied?"승급":"유지",reason(h.reason)])):"새 하루 승급전 결과 대기";
 const daily=(cycle.history||[]).slice().reverse().flatMap(row=>["champion","candidate"].flatMap(role=>Object.entries(row.accounts?.[role]||{}).map(([c,b])=>[row.session,role,c,accountPercent(b.net_return_rate),accountMoney(b.net_pnl,c),accountMoney(b.costs,c),whole(b.trade_count)])));
 $("dailyAccountHistory").innerHTML=daily.length?accountTable(["기록일","모델","통화","계좌 누적 수익률","누적 순손익","누적 비용","누적 체결"],daily):"첫 일일 판정 기록 대기 중입니다. 장기 계좌 누적 기록과 고정 시험 승부 점수는 구분합니다.";
}
function renderInferenceWork(d){
 const m=d.metrics||{},obs=d.account_observability||{},p=obs.candidate?.inference_profile||{},o=m.candidate_live_observation_profile||{},seconds=x=>x==null?"미측정":decimal(x,3)+"초";
 text("inferenceWorkSummary","Champion과 Candidate가 각자 추론·가상매매하고 양쪽 경험으로 각각 학습합니다. Champion "+whole(obs.champion?.inference_count)+"회 / Candidate "+whole(obs.candidate?.inference_count)+"회 · Candidate 미처리 관찰 "+whole(m.shared_observation?.pending)+"개. 학습 작업본과 승급전 고정본은 운영 계좌의 추론 모델과 구분합니다.");
 const rows=[
 ["Champion 실시간 판단",seconds(obs.champion?.last_inference_seconds),"입력 준비 후 forward · GPU 완료 대기 포함"],
 ["Candidate 관찰 처리 전체",seconds(o.total_seconds),"DB 읽기 → 체결·보상 → 추론 → 경험 저장 → 계좌 저장"],
 ["Candidate DB 관찰 읽기",seconds(o.read_seconds),"저장소 잠금 대기 포함"],
 ["Candidate 체결·보상 처리",seconds(o.fills_and_rewards_seconds),"성숙 경험을 한 transaction으로 기록"],
 ["Candidate 경험·계좌 DB 기록",seconds(o.database_commit_seconds),"관찰 완료와 pending 경험을 함께 저장"],
 ["Candidate 추론 경로 합계",seconds(p.total_seconds),"모델 잠금·GPU 대기 → forward → RAM 반환"],
 ["Candidate 추론 대기",seconds(p.wait_seconds),"GPU 우선순위 대기 + 가중치 잠금 대기"],
 ["Candidate 모델 GPU 이동",seconds(p.upload_seconds),"RAM → GPU"],
 ["Candidate 입력·forward",seconds(p.forward_seconds),"입력 준비 + forward + GPU 완료 대기"],
 ["Candidate 모델 RAM 반환",seconds(p.download_seconds),"GPU → RAM · VRAM 확보"],
 ["Candidate 관찰 가중치 갱신",seconds(m.candidate_observer_publish_seconds),m.candidate_observer_storage_reused?"기존 모델 저장공간 재사용":"최초 생성 / 측정 대기"]
 ];
 for(const role of ["champion","candidate"]){const r=m[role+"_last_completed_round"]||{};rows.push([role==="champion"?"Champion 최근 학습":"Candidate 최근 학습",seconds(r.compute_seconds)+" / 총 "+seconds(r.total_seconds),whole(r.unique_samples)+"개 경험 · optimizer "+whole(r.optimizer_steps)+"회"])}
  $("inferenceWorkDetails").innerHTML=accountTable(["작업","최근 경과 시간","측정 범위 / 처리량"],rows);
}
function renderOperationsOverview(d){
 const m=d.metrics||{},health=d.agent_health||{},candidate=health.candidate||{},schedule=m.gpu_scheduler||{};
 const names={champion_live:"Champion 판단",candidate_live:"Candidate 판단",champion_learning_step:"Champion 학습",candidate_learning_step:"Candidate 학습",champion_learning_setup:"Champion 학습 준비",candidate_learning_setup:"Candidate 학습 준비",candidate_publish:"Candidate 완료 가중치 반영",validation_champion:"Champion 승급전",validation_candidate:"Candidate 승급전"};
 const roleName=role=>names[role]||role||"작업 대기",lag=v=>v==null?"미측정":whole(v)+"초";
 text("opChampionLag",!d.agent_process_running?"정지":lag(health.lag_seconds));
 text("opChampionTime","완료 판단 "+timeOf(health.agent_cursor_timestamp_utc));
 text("opCandidateLag",!d.agent_process_running?"정지":lag(candidate.lag_seconds));
 text("opCandidateQueue","처리 대기 "+whole(candidate.pending??m.shared_observation?.pending)+"개 · 완료 "+timeOf(candidate.cursor_timestamp_utc));
 text("opGpuWork",!d.agent_process_running?"정지":schedule.policy?roleName(schedule.active):"새 계측 적용 대기");
 const learningFirst=schedule.policy==="preopen_replay_learning_first";
 text("opGpuQueue",schedule.policy?(learningFirst?"본장 전 학습 우선 · "+timeOf(schedule.learning_priority_until_utc)+"까지":"실시간 판단 우선")+" · 대기 "+((schedule.waiting||[]).map(x=>roleName(x.role)).join(" → ")||"없음"):"현재 실행 프로세스의 순서 계측 없음");
 text("opLearningQueue",whole(m.replay_eligible_backlog)+"개");
 text("opLastLearning","마지막 학습·저장 "+timeOf(m.last_update_utc));
 if(m.learning_wait_reason){
  text("dualLearningStatus","실시간 판단 먼저 처리 · replay 학습 대기 · 미학습 경험 보존");
  text("learningFlowHealth","시장 판단을 먼저 따라잡는 중입니다. 학습 회차는 현재 step 이후 대기하며 replay 경험은 보존합니다.");
  badge("learningState","판단 우선 · 학습 대기","warn");
  for(const role of ["Champion","Candidate"])text("dual"+role+"State","판단 우선 · 학습 대기");
 }
 if(learningFirst){
  text("dualLearningStatus","본장 전 replay 학습 우선 · 시세·미처리 관찰 보존");
  text("learningFlowHealth","20시 이후에는 두 모델의 미학습 경험을 우선 처리합니다. 판단·승급전은 남는 GPU 시간을 사용하며 본장 시작부터 판단 우선으로 복귀합니다.");
  badge("learningState","본장 전 학습 우선","blue");
 }
 const warnings=[];
 if(candidate.status==="error")warnings.push("Candidate 판단 오류: "+candidate.reason);
 else if(candidate.status==="stale")warnings.push("Candidate가 시세보다 "+lag(candidate.lag_seconds)+" 뒤처져 있습니다. 미처리 관찰 "+whole(candidate.pending)+"개는 DB에 남아 있으며 아래 Candidate 손익도 마지막 처리 시각 기준입니다.");
 if(health.status==="stale")warnings.push("Champion이 시세보다 "+lag(health.lag_seconds)+" 뒤처져 있습니다.");
 if(m.last_candidate_error)warnings.push("최근 Candidate 학습 오류: "+m.last_candidate_error);
 if(m.last_champion_error)warnings.push("최근 Champion 학습 오류: "+m.last_champion_error);
 $("runtimeAlert").hidden=!warnings.length;text("runtimeAlert",warnings.join(" "));
 // Use the same backend health and timestamps everywhere, rather than a
 // second client threshold based on feed-process wall-clock activity.
 if(d.agent_process_running){badge("agentBadge",health.status==="stale"?"Champion 판단 지연":health.status==="healthy"?"Champion 판단 정상":"판단 상태 확인",health.status==="stale"?"warn":"good");text("agentDetail","시세 "+timeOf(health.latest_feed_timestamp_utc)+" / 완료 판단 "+timeOf(health.agent_cursor_timestamp_utc)+" · "+lag(health.lag_seconds)+" 지연")}
}
render=function(d){
 renderBeforeAccountDiagnostics(d);renderAccountDiagnostics(d);renderDailyOperation(d);renderInferenceWork(d);renderOperationsOverview(d);
 const metrics=d.metrics||{},health=d.agent_health||{},candidate=health.candidate||{};
 const championDecision=metrics.champion_last_full_decision_timestamp||d.account_observability?.champion?.policy?.timestamp;
 const candidateDecision=d.account_observability?.candidate?.policy?.timestamp;
 text("opChampionTime","관찰 처리 "+timeOf(health.agent_cursor_timestamp_utc)+" · 실제 판단 "+timeOf(championDecision));
 text("opCandidateQueue","처리 대기 "+whole(candidate.pending)+"개 · 관찰 처리 "+timeOf(candidate.cursor_timestamp_utc)+" · 실제 판단 "+timeOf(candidateDecision));
 text("agentDetail","시세 "+timeOf(health.latest_feed_timestamp_utc)+" / 관찰 처리 "+timeOf(health.agent_cursor_timestamp_utc)+" / 실제 모델 판단 "+timeOf(championDecision));
 if(d.agent_process_running&&health.status==="healthy")badge("agentBadge","관찰 처리 정상","good");
 if(d.observe_enabled===false){
  badge("agentBadge","판단 OFF","");
  text("agentDetail","사용자가 새 모델 판단을 중지했습니다. 시세 저장·보유 평가·replay 학습은 계속됩니다.");
  for(const role of ["champion","candidate"]){
   badge(role+"LiveBadge","판단 OFF","");
   text(role==="champion"?"opChampionLag":"opCandidateLag","판단 OFF");
  }
  text("opChampionTime","추론 중지 · 시세 저장·보유 평가 계속");
  text("opCandidateQueue","추론 중지 · 기존 경험 처리·학습 계속");
 }else{
  const m=d.metrics||{},observer=d.candidate_live_account||{};
  if(m.champion_inference_skipped_reason==="context_only")badge("championLiveBadge","문맥 갱신 · 풀 추론 없음","");
  if(observer.status==="context_only")badge("candidateLiveBadge","문맥 갱신 · 풀 추론 없음","");
 }
};

const renderBeforeRuntimeUpdates=render;
render=function(d){
 renderBeforeRuntimeUpdates(d);
 let panel=document.getElementById("runtimeUpdatesPanel");
 if(!panel){
  panel=document.createElement("details");panel.id="runtimeUpdatesPanel";panel.className="details";
  const title=document.createElement("summary");title.textContent="설정·학습 코드 적용 상태 · 재시작 없이 갱신";panel.appendChild(title);
  const body=document.createElement("p");body.id="runtimeUpdatesBody";body.style.whiteSpace="pre-wrap";panel.appendChild(body);
  document.getElementById("runtimeDetails").appendChild(panel);
 }
 const m=d.metrics||{},u=m.runtime_updates,r=u&&u.applied_rules;
 const lines=["화면(JS/CSS): 새로고침 · 학습 규칙·loss/learner: 현재 회차 저장 후 hot apply",
  "모델 판단·관찰 Python: ‘모델 코드 적용’ 버튼으로 모델만 저장 후 재시작 · 시세 수집/웹 유지",
  "웹 API/server Python: 웹서버 재시작만 · feed/agent 유지"];
 if(r){
  lines.push(`현재 적용: batch ${r.training_batch_size} × optimizer ${r.training_optimizer_steps} · 결과 연결 ${r.reward_credit_seconds}초`);
  lines.push(`GPU 학습: ${m.champion_optimizer_backend||"확인 중"} / ${m.candidate_optimizer_backend||"확인 중"} · 손실 계산 ${m.candidate_loss_backend||m.champion_loss_backend||"확인 중"}`);
  if(u.applied_utc)lines.push("최근 적용: "+new Date(u.applied_utc).toLocaleString("ko-KR",{timeZone:"Asia/Seoul"}));
  const pending=Object.keys(u.deferred_rules||{});
  if(pending.length)lines.push("보류: "+pending.join(", ")+" · 진행 중 승급전 조건과 활성 계좌 목표·입력 구조는 별도 적용 필요");
  if(u.error)lines.push("적용 실패 · 기존 정상 값 유지: "+u.error);
 }else lines.push("실행 중 agent의 적용 상태를 확인 중입니다.");
 document.getElementById("runtimeUpdatesBody").textContent=lines.join("\n");
};
