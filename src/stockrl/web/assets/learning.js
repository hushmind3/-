function renderModelComparison(d){
 const m=d.metrics||{},v=d.validation_comparison||{},required=Number(v.bars_required)||390;
 const seconds=x=>x==null?"미측정":Number(x).toFixed(2)+"초",bytes=x=>x==null?"미측정":(Number(x)/1073741824).toFixed(2)+" GiB",score=x=>x==null?"미측정":(Number(x)*100).toFixed(4)+"%";
 for(const role of ["champion","candidate"]){
  const title=role==="champion"?"Champion":"Candidate",r=m[role+"_last_completed_round"]||{},candidate=role==="candidate";
  const params=candidate?m.candidate_total_parameter_count:m.parameters,trainable=m[role+"_trainable_parameter_count"];
  text("compare"+title+"Params",params==null?"미측정":whole(params)+"개");
  text("compare"+title+"Trainable",trainable==null?"미측정":whole(trainable)+"개");
  const enabled=m[role+"_learning_enabled"],training=m[role+"_training"];
  text("compare"+title+"LearningState",!d.agent_process_running?"정지 · 마지막 기록":d.learning_enabled===false?"학습 OFF · replay 보존":!enabled?"학습 비활성":training?"학습 중 · optimizer "+whole(m[role+"_optimizer_steps_current"])+"회":"학습 활성 · 회차 사이 대기");
  const changes=m[candidate?"weight_delta_l1":"champion_weight_delta_l1"],delta=Array.isArray(changes)&&changes.length?changes[changes.length-1]:null,version=r.model_version;
  text(candidate?"compareWeightDelta":"compareChampionWeightDelta",(version==null?"학습 버전 미기록":"최근 학습 완료 v"+whole(version))+" · L1 "+(delta==null?"미측정":Number(delta).toExponential(3)));
  text("compare"+title+"Compute",whole(m[role+"_live_inference_count"])+"회 · 경과 시간 누계 "+seconds(m[role+"_live_inference_seconds_total"]));
  text("compare"+title+"Learning",r.completed_utc?whole(r.unique_samples)+"개 경험 · optimizer "+whole(r.optimizer_steps)+"회 · 계산 "+seconds(r.compute_seconds)+" / 총 "+seconds(r.total_seconds)+" · "+timeOf(r.completed_utc):"완료 회차 미측정");
  text("compare"+title+"Gpu",r.completed_utc?"allocated "+bytes(r.peak_allocated_bytes)+" · reserved "+bytes(r.peak_reserved_bytes):"완료 회차 미측정");
 }
 const history=m.candidate_gate_history||[],finished=[...history].reverse().find(h=>
  h.candidate_version!=null&&h.champion_version!=null&&Number(h.required_bars)>=required&&Number(h.bars)>=Number(h.required_bars)&&
  h.candidate_score!=null&&h.champion_score!=null&&Number.isFinite(Number(h.candidate_score))&&Number.isFinite(Number(h.champion_score)));
 text("compareChampionScore",finished?score(finished.champion_score):"현재 기준 완료 기록 없음");
 text("compareCandidateScore",finished?score(finished.candidate_score):"현재 기준 완료 기록 없음");
 text("compareWinner",finished?timeOf(finished.time_utc)+" · 고정 Champion v"+whole(finished.champion_version)+" / Candidate v"+whole(finished.candidate_version)+" · "+(finished.applied?"Candidate 승급":"Champion 유지")+" · "+(Number(finished.candidate_score)>Number(finished.champion_score)?"Candidate 점수 우세":Number(finished.candidate_score)<Number(finished.champion_score)?"Champion 점수 우세":"동점"):"현재 최소 관측 기준을 마친 승급전 기록이 없습니다. 과거 운영 기록은 상세 이력에서 확인합니다.");
 const currentScore=role=>{const books=Object.values(v[role]||{});return books.length&&books.every(b=>Number.isFinite(b.net_return_rate))?books.reduce((sum,b)=>sum+b.net_return_rate,0):null};
 const progress=whole(v.bars_current)+" / "+whole(required)+"개 시장 분 · "+(d.observe_enabled===false&&v.active?"\uC2B9\uAE09\uC804 \uC77C\uC2DC\uC815\uC9C0 \u00B7 \uD310\uB2E8 OFF":v.active?"\uC9C4\uD589 \uC911 \u00B7 \uBBF8\uD655\uC815":"\uC2DC\uD5D8 \uBE44\uD65C\uC131 \u00B7 \uB9C8\uC9C0\uB9C9 \uC0C1\uD0DC");
 text("compareProgress",progress+" · 고정 Champion v"+whole(v.champion_snapshot_version)+" "+score(currentScore("champion"))+" / Candidate v"+whole(v.snapshot_version)+" "+score(currentScore("candidate"))+" · 계좌 시각 "+timeOf(v.last_timestamp));
 text("modelComparisonExplanation","두 모델 모두 판단·가상매매·학습합니다. 누적 횟수는 과거 실행 기록까지 포함해 같은 시작 시각의 대결 횟수가 아닙니다. 누적 판단 시간의 측정 범위도 다릅니다: Champion은 입력 준비 후 계산, Candidate는 대기·이동까지 포함합니다. 최근 학습은 양쪽 동일한 항목으로 표시합니다. VRAM은 해당 회차 중 전체 프로세스 최고치입니다. L1은 가중치 변화량이며 수익률이나 성장 점수가 아닙니다. 완료 점수는 고정본 버전과 관측 길이가 확인된 최근 승급전 기록이고, 진행 중 점수는 현재 시험 계좌에서 계산합니다.");
}
const baseRenderLearningMeasurements=renderLearning;
renderLearning=function(d){baseRenderLearningMeasurements(d);renderModelComparison(d);const m=d.metrics||{},l=d.learning||{},gb=v=>v==null?"—":(Number(v)/1073741824).toFixed(2)+" GB",pct=v=>v==null?"—":(Number(v)*100).toFixed(4)+"%",missing="아직 측정 전";text("paperTradeCount",whole(l.paper_trade_count)+"건 (KRW "+whole(l.paper_trade_counts_by_currency?.KRW)+" · USD "+whole(l.paper_trade_counts_by_currency?.USD)+")");text("currentValidationProgress",whole(l.validation_bars_current)+" / "+whole(l.validation_bars_required)+" bar");text("candidateTrainingSamples",l.candidate_training_samples==null?missing:whole(l.candidate_training_samples)+" / "+whole(l.candidate_training_samples_target)+" samples (고유 "+whole(l.candidate_training_unique_samples)+")");text("candidateOptimizerCount",l.candidate_optimizer_steps==null?missing:whole(l.candidate_optimizer_steps)+" / "+whole(l.candidate_optimizer_steps_target)+"회");text("candidateTrainingTime",l.candidate_update_seconds==null?missing:num(l.candidate_update_seconds).toFixed(1)+"초");text("candidateTrainingVram",l.candidate_peak_allocated_bytes==null?missing:"최고 "+gb(l.candidate_peak_allocated_bytes)+" · 시작 "+(l.candidate_baseline_allocated_bytes==null?"미측정":gb(l.candidate_baseline_allocated_bytes))+" · 예약 "+gb(l.candidate_peak_reserved_bytes));const currentGpuAllocated=l.gpu_allocated_bytes,currentGpuReserved=l.gpu_reserved_bytes;text("modelGpuMemoryCurrent",currentGpuAllocated==null?missing:gb(currentGpuAllocated)+" allocated / "+(currentGpuReserved==null?"n/a":gb(currentGpuReserved))+" reserved");const weightDeltas=m.weight_delta_l1;const lastDelta=Array.isArray(weightDeltas)&&weightDeltas.length?Number(weightDeltas[weightDeltas.length-1]):null;text("candidateWeightDelta",lastDelta==null?missing:lastDelta===0?"\ubcc0\ud654 \uc5c6\uc74c (0)":"\ubcc0\uacbd\ub428 (L1) "+lastDelta.toExponential(3));text("championWeightsBytes",gb(m.champion_model_parameter_bytes));text("championInferenceUsage",whole(m.champion_live_inference_count)+"회 · "+num(m.champion_live_inference_seconds_total).toFixed(1)+"초 · p50/p95 "+(num(m.inference_seconds_p50)*1000).toFixed(0)+"/"+(num(m.inference_seconds_p95)*1000).toFixed(0)+"ms");text("championValidationUsage",whole(m.champion_validation_inference_count)+"회 · "+num(m.champion_validation_inference_seconds_total).toFixed(1)+"초");text("candidateValidationUsage",whole(m.candidate_validation_inference_count)+"회 · "+num(m.candidate_validation_inference_seconds_total).toFixed(1)+"초");text("candidateScore",l.candidate_validation_score==null?"—":pct(l.candidate_validation_score));text("championScore",l.champion_validation_score==null?"—":pct(l.champion_validation_score));const phase=d.learning_enabled===false?"\ud559\uc2b5 OFF \u00b7 replay \ubcf4\uc874":l.candidate_stage==="sequential_paper_validation"?"\uac80\uc99d \ube44\uad50 \uc9c4\ud589 \uc911":l.candidate_stage==="training"?"candidate \ud559\uc2b5 \uc911":l.candidate_learning_enabled?"candidate \ud559\uc2b5 \ub300\uae30":"candidate \uc790\ub3d9 \ud559\uc2b5 \uaebc\uc9d0";const scores=l.candidate_validation_score==null||l.champion_validation_score==null?"\uc644\ub8cc\ub41c \ube44\uad50 \uc810\uc218 \uc5c6\uc74c":"candidate "+pct(l.candidate_validation_score)+" / champion "+pct(l.champion_validation_score);const gate=l.promotion_gate_ready?"\uc2b9\uae09 \ube44\uad50 \uc870\uac74 \ucda9\uc871":"\uc2b9\uae09 \ub300\uae30: "+(l.promotion_blocked_reason||l.candidate_skip_reason||"\uac80\uc99d \uc870\uac74 \ubbf8\ucda9\uc871");text("gateStory",phase+" \u00b7 "+scores+" \u00b7 "+gate+" \u00b7 \ube44\uc6a9 \ucc28\uac10 \uc21c\uc190\uc775\ub960");if(l.candidate_skip_reason)text("learningThreshold","검증 진행 "+whole(l.validation_bars_current)+" / "+whole(l.validation_bars_required)+" bar · 학습 표본 "+whole(l.replay_current)+" / "+whole(l.candidate_every)+" · 대기 이유: "+l.candidate_skip_reason)};
function renderLearningFlow(d){const m=d.metrics||{},h=d.agent_health||{},candidateHealth=h.candidate||{},lag=value=>value==null?(!d.agent_process_running?"정지":"미측정"):num(value).toFixed(0)+"초",training=!!m.candidate_training,trial=!!m.candidate_validation_active,trialPaused=trial&&d.observe_enabled===false;const progress=trialPaused?"\uc2b9\uae09\uc804 \uc77c\uc2dc\uc815\uc9c0 \u00b7 \uc2dc\uc7a5 \uad00\ucc30 OFF":training?"Candidate 학습 중 · optimizer "+whole(m.candidate_optimizer_steps_current)+" / "+whole(m.candidate_optimizer_steps_target)+"회 · "+whole(m.candidate_samples_current)+"개 replay 입력 처리":m.candidate_skip_reason||"새 경험 대기";text("learningFlowHealth",progress+" | 남은 학습 가능한 replay "+whole(m.candidate_eligible_replay_count??m.candidate_untrained_replay_count)+"건 | 검증 "+whole(m.candidate_validation_bars)+" / "+whole(m.candidate_min_validation_dates||390)+"개 시장 분 · 대기열 "+whole(m.candidate_validation_queue_depth)+" / "+whole(m.candidate_validation_queue_capacity)+" | Champion feed 대비 커서 차이 "+lag(h.lag_seconds)+" · Candidate feed 대비 커서 차이 "+lag(candidateHealth.lag_seconds));badge("candidateBadge",trialPaused?"\uC2B9\uAE09\uC804 \uC77C\uC2DC\uC815\uC9C0":training&&trial?"학습 + 고정 시험본 검증":training?"Candidate 학습 중":trial?"고정 시험본 검증 중":"경험 대기",training?"blue":trial?"good":"");text("candidateOptimizerCount",whole(training?m.candidate_optimizer_steps_current:m.last_candidate_optimizer_steps)+" / "+whole(m.candidate_optimizer_steps_target)+"회");text("candidateTrainingSamples",whole(training?m.candidate_samples_current:m.last_candidate_samples_trained)+"개 · 목표 "+whole(m.candidate_samples_target)+"개");const round=m.candidate_last_completed_round;text("candidateTrainingTime",round?"\uCD5C\uADFC \uC644\uB8CC: \uCD1D "+num(round.total_seconds).toFixed(1)+"\uCD08 \u00B7 \uACC4\uC0B0 "+num(round.compute_seconds).toFixed(1)+"\uCD08 \u00B7 \uB2E8\uACC4\uB2F9 "+num(round.step_compute_seconds).toFixed(1)+"\uCD08":m.candidate_training?"\uD559\uC2B5 \uC911 \u00B7 \uC774\uBC88 \uD68C\uCC28 \uC644\uB8CC \uD6C4 \uC2DC\uAC04 \uCE21\uC815":"\uC544\uC9C1 \uCE21\uC815 \uC804");const coverage=m.multiscale_coverage||{};text("appliedSettings","최근 입력 기록 확보: "+Object.entries(coverage).map(([k,v])=>({"1m":"1분","3m":"3분","5m":"5분","15m":"15분","60m":"60분","1d":"일","1w":"주","1mo":"월"}[k]||k)+" "+(Number(v)*100).toFixed(0)+"%").join(" · ")+" · 매수/매도 호가 제공 "+whole(m.quoted_bid_ask_symbols)+"\uC885\uBAA9 \u00B7 \uB9C8\uC9C0\uB9C9 \uC804\uCCB4 \uCD94\uB860 \uAE30\uC900 "+(m.multiscale_input_status_utc?timeOf(m.multiscale_input_status_utc):"\uBBF8\uCE21\uC815"));const b=d.backtest||{};text("portfolioBacktestSummary",b.evaluation_mode?"\uCD5C\uADFC \uACC4\uC88C \uBC31\uD14C\uC2A4\uD2B8: "+whole(b.timestamps)+"\uAD6C\uAC04 \u00B7 \uCCB4\uACB0 "+whole(b.trade_count)+"\uAC74 \u00B7 \uC21C\uC790\uC0B0 \uC218\uC775\uB960 "+(num(b.net_return)*100).toFixed(4)+"% \u00B7 "+num(b.elapsed_seconds).toFixed(1)+"\uCD08. \uACFC\uAC70 \uC9C4\uB2E8\uC774\uBA70 \uC2B9\uAE09 \uD310\uC815\uC5D0\uB294 \uC0AC\uC6A9\uD558\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.":"\uACC4\uC88C \uBC31\uD14C\uC2A4\uD2B8 \uAE30\uB85D \uC5C6\uC74C");const error=m.agent_last_input_error||m.last_candidate_error||m.candidate_validation_error||m.candidate_live_error;if(error){text("learningFlowHealth",$("learningFlowHealth").textContent+" | 확인할 오류: "+error)}text("gateStory",trialPaused?"\uC2B9\uAE09\uC804 \uC77C\uC2DC\uC815\uC9C0 \u00B7 \uC2DC\uC7A5 \uAD00\uCC30 OFF":trial?"고정 시험본의 새로운 미래 구간 비교 "+whole(m.candidate_validation_bars)+" / 390 · 동일 자금·비용의 순자산 성과로 판정":m.candidate_validation_error?"최근 시험 무효: "+m.candidate_validation_error:"새 미래 구간 시험 준비 · 마지막 완료 점수는 위 비교표 참조")}
function renderDailyLearning(d){
  const m=d.metrics||{},rows=m.daily_learning||[],host=$("dailyLearningRows");
  const objective=m.shared_objective;
  if(objective){
   const goals=["champion","candidate"].flatMap(role=>{
    const goal=m[role+"_goal"]||{};
    return Object.entries(goal.books||{}).map(([currency,b])=>[
     role==="champion"?"Champion":"Candidate",currency,
     num(b.multiple).toFixed(4)+"배",accountMoney(b.target_equity,currency),
     accountPercent(b.target_asset_ratio??(num(b.multiple)/num(goal.target_multiple||10))),accountPercent(b.net_return_rate??(num(b.multiple)-1)),b.win?"WIN · "+timeOf(b.win.timestamp):"진행 중"]);
   });
   const goalTraining=["champion","candidate"].map(role=>{
    const round=m[role+"_last_completed_round"],label=role==="champion"?"Champion":"Candidate";
    return label+": "+(round?.goal_conditioned_samples==null?"새 목표 입력의 완료 학습 회차는 아직 미확인":
     "최근 완료 회차 "+whole(round.goal_conditioned_samples)+" / "+whole(round.samples)+"개 목표 입력 학습 · 성공 보너스 경험 "+whole(round.goal_bonus_samples)+"개");
   }).join(" | ");
   $("sharedGoalSummary").innerHTML="<strong>공통 임무: 비용 차감 순자산 "+esc(num(objective.target_multiple).toFixed(0))+"배 · 두 모델 모두 순손익 + 목표 달성 경험으로 학습</strong><br>장기 계좌는 매일 유지합니다. 각 통화의 목표 달성은 episode당 한 번 기록하고 "+esc(whole(objective.win_bonus_points))+"점의 학습용 성공 보상을 연결합니다. 승부 점수는 실제 계좌 성과이며 성공 보너스는 포함하지 않습니다."+
    "<br>목표 자산 대비 달성률 = 현재 순자산 ÷ 목표 순자산입니다. 학습률과 다릅니다. 10배 목표에서는 시작 자금이 10%이며 손실은 원금 대비 수익률에 표시합니다."+
    (goals.length?accountTable(["모델","통화","현재 자산 배율","목표 순자산","목표 자산 대비 달성률","원금 대비 수익률","목표 상태"],goals):"<br>새 목표 방식 관측 대기")+"<br>"+esc(goalTraining);
  }
 const credit=m.reward_credit;
 if(credit){
  const scores=["champion","candidate"].map(role=>{
   const score=m[role+"_reward_score"],p=score?.points||{};
   return (role==="champion"?"Champion":"Candidate")+" 누적점수: "+(score?"KRW "+num(p.KRW).toFixed(3)+"점 · USD "+num(p.USD).toFixed(3)+"점":"새 방식 관측 대기");
  }).join(" | ");
  text("rewardCreditDescription","비용 차감 계좌 수익률 +1% = +1점, −1% = −1점. "+scores+". 새 보상은 직전 점수의 변화분입니다. 새 경험은 "+num(credit.duration_seconds/60).toFixed(0)+"분 동안 관측한 누적 결과와 다음 상태의 예상 가치를 연결합니다. 입력봉 개수나 강제 보유시간 제한이 아닙니다. 초기화로 이전 계좌의 연결을 종료하며, 기존 짧은 보상 경험도 학습 후 삭제합니다.");
 }
 const todayKey=new Intl.DateTimeFormat("sv-SE",{timeZone:"Asia/Seoul",year:"numeric",month:"2-digit",day:"2-digit"}).format(new Date()),today=rows.find(row=>row.day===todayKey)||{};
 text("todayLearningGenerated",whole(today.enqueued));text("todayLearningCompleted",whole(today.completed));text("todayLearningRemaining",whole(today.remaining));text("allLearningRemaining",whole(m.replay_eligible_backlog));text("blockedLearningCount",whole(Number(m.replay_quarantined_count||0)+Number(m.replay_unsupported_count||0)));
 const rate=Number(today.enqueued)>0?100*Number(today.completed||0)/Number(today.enqueued):null;
 text("learningCompletionSummary",rate==null?"오늘 아직 보상 평가가 끝나 새로 저장된 경험이 없습니다.":"오늘 생성 "+whole(today.enqueued)+"건 → 학습·저장 완료 "+whole(today.completed)+"건 ("+rate.toFixed(2)+"%) · 학습 가능 잔여 "+whole(today.remaining)+"건 · 보류 "+whole(today.blocked)+"건. 완료 경험은 DB에서 삭제되어 현재 파일 건수와 누계가 다릅니다.");
 text("pendingLearningCount",whole(m.replay_pending_count??m.pending_experiences)+"건");
 const blockedReasonNames={"saved portfolio input shape is incompatible":"\uc800\uc7a5\ub41c \uc785\ub825 \ud615\ud0dc \ubd88\uc77c\uce58 (\uc138\ubd80 \uc6d0\uc778 \uae30\ub85d \uc5c6\uc74c)","unsupported reward schema":"지원하지 않는 과거 보상 형식"};
 text("blockedLearningReason",Object.entries(m.replay_blocked_reasons||{}).map(([reason,count])=>(reason.startsWith("saved portfolio input shape is incompatible:")?"\uc800\uc7a5 \uc785\ub825 \ud615\ud0dc \ubd88\uc77c\uce58 \u00b7 "+reason.split(":").slice(1).join(":").trim():(blockedReasonNames[reason]||reason))+" "+whole(count)+"건").join(" · ")||"새 원인 계측 대기");
 text("completedReplayRetained",whole(m.replay_completed_retained));
 text("learningObservationTime",timeOf(m.last_market_timestamp)+" · "+(d.agent_health?.lag_seconds==null?"지연 미확인":num(d.agent_health.lag_seconds).toFixed(0)+"초"));
 text("learningDatabaseLocation",(m.replay_database_path||"경로 확인 중")+" · "+(Number(m.replay_file_bytes||0)/1048576).toFixed(2)+" MiB");
 host.replaceChildren();
 for(const day of rows){const tr=document.createElement("tr");for(const value of [day.day,whole(day.enqueued),whole(day.first_trained),whole(day.completed),whole(day.remaining)]){const td=document.createElement("td");td.textContent=value;tr.appendChild(td)}host.appendChild(tr)}
 if(!rows.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=5;td.textContent="경험 생성 대기";tr.appendChild(td);host.appendChild(tr)}
 const bytes=Number(m.replay_file_bytes||0),blocked=Number(m.replay_quarantined_count||0)+Number(m.replay_unsupported_count||0),forwards=m.candidate_training?m.candidate_window_forwards_current:m.last_candidate_window_forwards;
 text("dailyLearningSummary","학습 가능한 남은 경험 "+whole(m.replay_eligible_backlog)+"건 · 양쪽 학습 시작 미완료 "+whole(m.replay_untrained_count)+"건 · 학습 불가·보류 "+whole(blocked)+"건 · replay "+(bytes/1048576).toFixed(2)+" MiB · 용량 때문에 삭제하지 않음 · 최근 공유 입력 계산 "+whole(forwards)+"회");
 if(m.replay_oldest_unfinished_timestamp)text("dailyLearningSummary",$("dailyLearningSummary").textContent+" · 가장 오래된 남은 경험 "+timeOf(m.replay_oldest_unfinished_timestamp));
}
const previousRenderLearningFlow=renderLearningFlow;
function renderDualLearning(d){
 const m=d.metrics||{};
 const activeRole=m.champion_training?"champion":m.candidate_training?"candidate":null,backlog=Number(m.replay_eligible_backlog||0),passes=m.candidate_replay_passes,learningOn=d.learning_enabled!==false;
 const status=!d.running?"시스템 정지 · 저장된 마지막 학습 상태":!learningOn?"replay 학습 중지 · 미학습 "+whole(backlog)+"건 보존":activeRole?(activeRole==="champion"?"Champion":"Candidate")+" 학습 중 · 다른 모델은 다음 회차 대기":backlog?"학습 가능한 경험 "+whole(backlog)+"건 · 다음 회차 준비":"학습 가능한 미학습 0건 · 새 결과 경험 대기";
 text("dualLearningStatus",status);
 badge("learningState",!d.running?"정지":!learningOn?"학습 OFF":activeRole?"학습 중":backlog?"회차 준비":"새 경험 대기",!d.running?"bad":activeRole?"blue":"");
 badge("candidateBadge",!d.running?"정지":!learningOn?"학습 OFF · 경험 보존":activeRole?"두 모델 순차 학습":backlog?"학습 회차 준비":m.candidate_validation_active&&d.observe_enabled===false?"\uC2B9\uAE09\uC804 \uC77C\uC2DC\uC815\uC9C0 \u00B7 \uD310\uB2E8 OFF":m.candidate_validation_active?"\uBBF8\uD559\uC2B5 0 \u00B7 \uB300\uACB0 \uC9C4\uD589":"\uC0C8 \uACBD\uD5D8 \uB300\uAE30",activeRole?"blue":"");
 const health=d.agent_health||{},candidateHealth=health.candidate||{},lag=value=>value==null?(!d.agent_process_running?"정지":"미측정"):num(value).toFixed(0)+"초";
 text("learningFlowHealth",status+(activeRole?" · optimizer "+whole(m[activeRole+"_optimizer_steps_current"])+" / "+whole(m[activeRole+"_optimizer_steps_target"])+"회":"")+" | 평가 대기 "+whole(m.replay_pending_count??m.pending_experiences)+"건 | 대결 "+whole(m.candidate_validation_bars)+" / "+whole(m.candidate_min_validation_dates||390)+"개 시장 분 | Champion 지연 "+lag(health.lag_seconds)+lag(candidateHealth.lag_seconds));
 for(const role of ["champion","candidate"]){
  const title=role==="champion"?"Champion":"Candidate",active=!!m[role+"_training"],prefix="dual"+title;
  text(prefix+"State",!d.running?"정지":!learningOn?"학습 OFF · "+whole(m[role+"_eligible_replay_count"]||0)+"건 보존":active?"학습 중":Number(m[role+"_eligible_replay_count"])?"학습 회차 대기":"남은 경험 완료 · 새 경험 대기");
  text(prefix+"Version",whole(m[role==="champion"?"champion_training_version":"candidate_model_version"]));
  text(prefix+"Runs",whole(m[role+"_completed_training_runs"])+"회");
  text(prefix+"Exposures",whole(role==="champion"?m.champion_paper_examples_trained:m.paper_examples_trained)+"개 paper 경험");
  text(prefix+"Samples",whole(m[active?role+"_samples_current":"last_"+role+"_samples_trained"])+" / "+whole(m[role+"_samples_target"])+"개");
  text(prefix+"Steps",whole(m[active?role+"_optimizer_steps_current":"last_"+role+"_optimizer_steps"])+" / "+whole(m[role+"_optimizer_steps_target"])+"회");
  const completed=m[role+"_last_completed_round"];
  text(prefix+"CompletedAt",timeOf(completed?.completed_utc));
  text(prefix+"Time",completed?num(completed.total_seconds).toFixed(1)+" / "+num(completed.compute_seconds).toFixed(1)+"초":"완료 회차 측정 대기");
  const vram=completed?.peak_allocated_bytes;text(prefix+"Vram",vram==null?"측정 대기":(Number(vram)/1073741824).toFixed(2)+" GiB");
  text(prefix+"Remaining",whole(m[role+"_eligible_replay_count"])+"건");
 }
 text("compareChampionTrainable",whole(m.champion_trainable_parameter_count)+"개");
 const policy=passes==null?"학습 반복 횟수 미확인":"Champion과 Candidate가 같은 경험을 각각 "+whole(passes)+"회 학습하고 저장합니다. 둘 다 끝난 경험만 삭제합니다.";
 text("learningThreshold",policy);text("singlePassPolicy",policy+" 같은 GPU에서 회차를 번갈아 처리하며 서로 다른 시각의 경험과 모든 시간봉 입력은 유지합니다.");
 const candidateRound=m.candidate_last_completed_round;
 if(candidateRound)text("candidateTrainingTime","최근 완료: 총 "+num(candidateRound.total_seconds).toFixed(1)+"초 · 계산 "+num(candidateRound.compute_seconds).toFixed(1)+"초 · 단계당 "+num(candidateRound.step_compute_seconds).toFixed(1)+"초");
 text("timeframeMeasurementAt", "\uc785\ub825 \ud655\ubcf4\uc728 \uae30\uc900: "+(m.multiscale_input_status_utc?timeOf(m.multiscale_input_status_utc):"\ubbf8\uce21\uc815"));
 const host=$("timeframeLearningRows");host.replaceChildren();
 text("modelUniverseSummary","등록 "+whole(d.configured_instruments)+"종목 · 최근 5분 수신 "+whole((d.feed_metrics?.fresh_symbols_5m||[]).length)+"종목 · 실제 판단 입력 "+(m.model_input_symbol_count==null?"측정 대기":whole(m.model_input_symbol_count)+"종목")+" · 빈 입력 칸 "+whole(m.model_padding_symbol_count)+"개 · 종목 ID 미연결 "+whole(m.unmatched_live_symbol_count)+"종목 · 모델 종목 ID 공간 "+(m.model_symbol_id_capacity==null?"미확인":whole(m.model_symbol_id_capacity)+"개")+" (동시 입력 가능 수 보장 아님)");
 const names={"1m":"1분봉","3m":"3분봉","5m":"5분봉","15m":"15분봉","60m":"60분봉","1d":"일봉","1w":"주봉","1mo":"월봉"};
 for(const [scale,label] of Object.entries(names)){
  const info=m.multiscale_input_status?.[scale],coverage=info?.mean_history_coverage??m.multiscale_coverage?.[scale],c=m.champion_last_completed_round?.timeframe_samples?.[scale],a=m.candidate_last_completed_round?.timeframe_samples?.[scale];
  const values=[label,coverage==null?"측정 대기":(Number(coverage)*100).toFixed(1)+"%",info?whole(info.available_symbols)+" / "+whole(info.observed_symbols):"측정 대기",info?whole(info.complete_history_symbols)+"종목":"측정 대기",c==null?"측정 대기":whole(c)+" / "+whole(m.champion_last_completed_round.samples)+"개",a==null?"측정 대기":whole(a)+" / "+whole(m.candidate_last_completed_round.samples)+"개"];
  const tr=document.createElement("tr");for(const value of values){const td=document.createElement("td");td.textContent=value;tr.appendChild(td)}host.appendChild(tr);
 }
 text("replayDetail","학습 가능 "+whole(m.replay_eligible_backlog)+"건 · 보류 "+whole(Number(m.replay_quarantined_count||0)+Number(m.replay_unsupported_count||0))+"건 · 현재 DB 잔여");
 const inputError=m.agent_last_input_error||m.last_champion_error||m.last_candidate_error||m.candidate_validation_error;if(inputError)text("learningFlowHealth",$("learningFlowHealth").textContent+" | 오류: "+inputError);
 const champError=m.last_champion_error;if(champError)text("dualLearningStatus",$("dualLearningStatus").textContent+" · Champion 오류: "+champError);
}
renderLearningFlow=function(d){previousRenderLearningFlow(d);renderDailyLearning(d);renderDualLearning(d)};
