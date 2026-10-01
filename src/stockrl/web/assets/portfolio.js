function accountMoney(value,currency){return value==null?"—":Number(value).toLocaleString("ko-KR",{maximumFractionDigits:currency==="KRW"?0:2})+" "+currency}
function accountPercent(value){return value==null?"—":(Number(value)*100).toFixed(2)+"%"}
function accountTable(headers,rows){return '<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function renderAccountDiagnostics(d){
 const view=d.account_observability;
 if(!view){for(const role of ["champion","candidate"])text(role+"AccountOverview","신규 계좌 지표 연결 대기");return}
 for(const role of ["champion","candidate"]){
  const account=view[role]||{},books=account.books||{},currentPolicy=account.policy||{},isChampion=role==="champion";
  const health=isChampion?d.agent_health:d.agent_health?.candidate;
  const active=Boolean(d.agent_process_running),enabled=Boolean(d.paper_enabled);
  const badge=isChampion?"championLiveBadge":"candidateLiveBadge";
  const error=account.observer_error||health?.status==="error";
  text(badge,error?"판단 오류":!active?"정지 · 마지막 기록":d.observe_enabled===false?"판단 OFF · 계좌 평가 유지":health?.status==="stale"?"시세보다 "+Math.ceil(num(health.lag_seconds)/60)+"분 지연":enabled?"판단·가상매매":"관찰만");
  $(badge).className="pill "+(error?"danger":health?.status==="stale"?"warn":active&&health?.status==="healthy"?"good":"");
  const currentHasTrades=num(currentPolicy.observed_tradable_symbols)>0;
  const policy=currentHasTrades?currentPolicy:account.last_tradable_policy||{};
  $(role+"AccountOverview").innerHTML=["KRW","USD"].map(currency=>{
   const b=books[currency];if(!b)return '<div class="account-currency">'+currency+' · 계좌 대기</div>';
   const tone=num(b.net_pnl)<0?"account-negative":"account-positive";
   return '<div class="account-currency"><div class="currency-title">'+currency+' · 비용 차감 수익률</div><strong class="'+tone+'">'+esc(accountPercent(b.net_return_rate))+'</strong><div class="net-value '+tone+'">'+esc(accountMoney(b.net_pnl,currency))+'</div><small>현재 순자산 '+esc(accountMoney(b.equity,currency))+'<br>현금 '+esc(accountPercent(b.cash_ratio))+' · 최대 종목 비중 '+esc(accountPercent(b.largest_position_weight))+'<br>'+(!num(b.trade_count)?'아직 체결 없음':'누적 '+whole(b.trade_count)+'건 체결')+'</small></div>';
  }).join('');
  const pair=key=>["KRW","USD"].map(c=>accountMoney(books[c]?.[key],c)).join(" · ");
  text(isChampion?"championLiveSeed":"candidateLiveSeed",pair("initial_cash"));
  text(isChampion?"championLiveCash":"candidateLiveCash",pair("cash"));
  text(isChampion?"paperCosts":"candidateLiveCosts",pair("costs"));
  text(isChampion?"paperTradeCount":"candidateLiveTrades",["KRW","USD"].map(c=>c+" "+whole(books[c]?.trade_count)+"건").join(" · "));
  text(isChampion?"championLivePositionCount":"candidateLivePositionCount",["KRW","USD"].map(c=>c+" "+whole(books[c]?.position_count)+"종목").join(" · "));
  const readings=[];
  for(const currency of ["KRW","USD"]){const b=books[currency];if(!b)continue;
   if(!num(b.trade_count)){readings.push(currency+"은 체결이 없어 아직 매매 성과를 평가할 수 없습니다.");continue}
   readings.push(currency+" 순손익 "+accountMoney(b.net_pnl,currency)+", 누적 비용 "+accountMoney(b.costs,currency)+". 기록된 체결 손익에 비용만 더하면 "+accountMoney(b.recorded_pnl_plus_costs,currency)+"입니다.");
   if(!b.reconciliation_ok)readings.push(currency+" 손익 합산 불일치: "+accountMoney(b.reconciliation_difference,currency));
  }
  text(role+"AccountReading",readings.join(" ")+" 비용을 더한 값은 무비용 재실험 결과가 아닙니다.");
  const status=account.observer_error?"관찰 오류: "+account.observer_error:account.training?"학습 중 · 저장된 최신 가중치로 관찰":"저장된 가중치로 관찰";
  const profile=account.inference_profile||mInferenceProfile(d,role),queue=isChampion?0:num(d.metrics?.shared_observation?.pending);
  const part=v=>v==null?"미측정":decimal(v,2)+"초";
  $(role+"AccountMeta").className="account-meta"+(health?.status==="stale"?" delayed":"");
  text(role+"AccountMeta","마지막 실제 모델 판단 "+timeOf(account.last_full_decision_timestamp)+" · 계좌 평가·관찰 처리 "+timeOf(account.last_timestamp)+" · 시세 대비 "+(health?.lag_seconds==null?"지연 미측정":whole(health.lag_seconds)+"초 지연")+" · 관측 가중치 v"+whole(account.version)+". 최근 GPU 대기 "+part(profile.wait_seconds)+" / 입력·계산 "+part(profile.forward_seconds)+(isChampion?"":" · 미처리 관찰 "+whole(queue)+"개"));
  const positions=["KRW","USD"].flatMap(c=>(books[c]?.positions||[]).map(p=>[instrumentLabel(p.symbol,d),c,whole(p.quantity)+"주",accountMoney(p.average_cost,c),accountMoney(p.mark,c)+(p.mark_available?"":" (평단 대체)"),accountPercent(p.weight),accountMoney(p.unrealized_pnl,c)]));
  $(isChampion?"positions":"candidateLiveHoldings").innerHTML=positions.length?accountTable(["종목","통화","수량","평단","평가가격","계좌 비중","평가손익"],positions):"보유 종목 없음";
  const costs=["KRW","USD"].filter(c=>books[c]).map(c=>{const b=books[c];return [c,accountMoney(b.fees,c),accountMoney(b.sell_tax,c),accountMoney(b.spread,c),accountMoney(b.slippage,c),accountPercent(b.cost_return_rate),b.reconciliation_ok?"일치":"불일치"]});
  $(role+"CostDetails").innerHTML=accountTable(["통화","수수료","세금","스프레드","슬리피지","시드 대비 비용","순손익 = 실현 + 평가"],costs);
  $(role+"TradingStatistics").innerHTML=accountTable(["통화","통계 시작","기록 매수 / 매도","자연시간당 체결","매도 체결 승률","평균 매도 순손익","평균 / 최단 / 최장 보유"],["KRW","USD"].filter(c=>books[c]).map(c=>{
   const stats=books[c].trade_statistics||{},seconds=value=>value==null?"미측정":decimal(Number(value)/60,1)+"분";
   return [c,stats.first_timestamp?timeOf(stats.first_timestamp):"새 체결 대기",whole(stats.buy_count)+" / "+whole(stats.sell_count),decimal(stats.fills_per_elapsed_hour,1),accountPercent(stats.sell_win_rate),accountMoney(stats.mean_sell_net_pnl,c),[stats.mean_holding_seconds,stats.holding_seconds_min,stats.holding_seconds_max].map(seconds).join(" / ")];
  }))+"<p class=\"help\">통계 기록을 시작한 이후의 체결만 집계합니다. 승률은 비용을 뺀 매도 체결별 손익 기준이며 부분매도도 한 건입니다. 보유시간은 최초 진입에서 해당 매도까지이고, 과거 진입시각이 없는 보유분은 시간 집계에서 제외합니다.</p>";
  const decisions=policy.decisions||[];
  const counts=policy.action_counts||{};
  const policyAsOf=currentHasTrades?"이번 관측 ":"지금 새 거래 가능 시세는 0종목입니다. 마지막 거래 가능 관측 기록 ";
  text(role+"PolicySummary",decisions.length?policyAsOf+timeOf(policy.timestamp)+" · "+whole(policy.observed_tradable_symbols)+"종목: 매수 "+whole(counts.BUY)+" / 관망 "+whole(counts.HOLD)+" / 매도 "+whole(counts.SELL)+" · 새 주문 "+whole(policy.submitted_orders)+"건 · 미보유 매도 "+whole(policy.sell_without_position)+"건. 평균 최고 확률 "+accountPercent(policy.mean_top_probability)+", 확률 퍼짐 "+accountPercent(policy.mean_normalized_entropy)+" (100%는 세 행동의 확률이 같음). 최고 확률과 다른 추첨 "+whole(policy.sampled_actions_different_from_argmax)+"건. 현금 목표 KRW "+accountPercent(policy.effective_cash_target?.KRW)+" / USD "+accountPercent(policy.effective_cash_target?.USD)+". 목표 비중은 주문 계산에 쓰는 값이며 현재 보유 비중과 다릅니다.":"새 거래 가능 시세의 정책 지표 수집 대기 · 오래된 판단을 현재 값으로 대신 표시하지 않습니다.");
  $(isChampion?"modelDirections":"candidateLiveDirections").innerHTML=decisions.length?accountTable(["종목","판단","매도 확률","관망 확률","매수 확률","목표 비중","보유량","새 주문"],decisions.map(row=>[instrumentLabel(row.symbol,d),{BUY:"매수",HOLD:"관망",SELL:"매도"}[row.action]||row.action,accountPercent(row.p_sell),accountPercent(row.p_hold),accountPercent(row.p_buy),accountPercent(row.allocation_target),whole(row.held_quantity)+"주",row.order_submitted?"제출":"없음"])):"관측 대기";
  const fills=(account.recent_fills||[]).slice().reverse();
  $(isChampion?"championLiveFills":"candidateLiveFills").innerHTML=fills.length?accountTable(["시각","종목","체결","수량","체결가","수수료","매도세"],fills.map(f=>[timeOf(f.date),instrumentLabel(f.symbol,d),f.action==="BUY"?"매수":"매도",whole(f.quantity)+"주",accountMoney(f.price,f.currency),accountMoney(f.fee,f.currency),accountMoney(f.sell_tax,f.currency)])):"체결 없음";
 }
 text("accountInputScope","1·3·5·15·60분봉 + 일·주·월봉 요약을 함께 입력합니다. 저장된 마지막 시세 중 최우선 매수·매도 호가가 있는 종목 "+whole(view.quoted_bid_ask_symbols)+"개 (현재 수신 여부는 시장 수신 상태 참조). 전체 호가 단계는 미수집입니다. USD 비용 가정: 편도 수수료 "+accountPercent(view.fee_rate)+", 슬리피지 "+decimal(view.slippage_bps,1)+"bp. 왕복 비용은 약 "+accountPercent(view.usd_round_trip_cost_rate_before_spread)+"부터이며 스프레드는 별도입니다. 한국 매도세는 "+accountPercent(view.krw_sell_tax_assumption)+"의 모의 가정입니다. 실제 주문 OFF.");
  text("accountRewardHorizon","현재 새 학습 경험은 "+num(d.metrics?.reward_credit?.duration_seconds/60).toFixed(0)+"분 동안 결과를 연결하고 다음 상태의 예상 가치도 사용합니다. 시간봉 입력 길이와 별개입니다. 기존 짧은 경험은 이전 방식 그대로 학습하며, 미보유 매도·미체결 판단을 체결로 세지 않습니다.");
}
function mInferenceProfile(d,role){return d.metrics?.[role+"_live_inference_profile"]||{}}
