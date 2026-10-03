import { api } from '../api';
import { usePolling } from '../hooks';
import { bytes, fmt, list, num, obj, pct, str, time } from '../data';
import { FillHistory } from '../components/operations';
import { ActionButton, DataTable, ErrorState, JsonDetails, Loading, Signal, StageTrack } from '../components/ui';

export function TradingMoE() {
  const q=usePolling('/api/trading-moe/status',1000), r=q.data;
  if(!r)return q.error?<ErrorState message={q.error} retry={q.refresh}/>:<Loading/>;
  const decision=obj(r.decision), learning=obj(r.learning), compute=obj(r.compute), stages=obj(r.stages), replay=obj(r.replay), live=!!r.alive,
    books=Object.entries(obj(r.books)).map(([currency,value])=>{const book=obj(value), pnl=num(book.net_pnl??(num(book.equity)-num(book.initial_cash)));return[currency,fmt(book.equity)+' '+currency,fmt(book.cash),fmt(pnl)+' · '+pct(book.net_return_rate??(num(book.initial_cash)?pnl/num(book.initial_cash):0)),fmt(book.trade_count,0)+'회',fmt(book.costs??(num(book.fees)+num(book.slippage)+num(book.spread)+num(book.sell_tax)))];});
  return <>
    {q.error&&<ErrorState message={q.error} retry={q.refresh}/>}
    <section className="moe-decision"><div className="moe-command"><div className="moe-worker-state"><small>INDEPENDENT WORKER · PAPER</small><Signal label="TradingMoE" value={r.gpu_waiting?'GPU 차례 대기':str(r.status)} tone={r.error?'bad':live?'good':'neutral'}/><span>PID {str(r.pid)} · 적재 {fmt(r.load_count,0)}회</span><span>{str(r.checkpoint)}</span></div><div className="action-row"><ActionButton label="TradingMoE 시작" path={api.moe('start')} disabled={live||r.status==='loading'}/><ActionButton label="TradingMoE 저장 후 정지" path={api.moe('stop')} disabled={!live} pendingLabel="저장 · 정지 요청 중…"/></div>{r.error&&<p className="error-text">{str(r.error)}</p>}</div>
      <div className="decision-head"><div><small>최종 controller 결정 · {time(decision.as_of||r.market_timestamp)}</small><strong>ETHUSDT <em className={'action-word '+str(decision.action,'')}>{str(decision.action,'대기')}</em></strong><span>현재 비중 {pct(decision.current_weight)} <b>→</b> 목표 {pct(decision.target_weight)} · 현금 목표 {pct(decision.cash_weight)}</span></div><div className="decision-clock"><small>의사결정 소요</small><strong>{fmt(decision.seconds,3)}초</strong><small>전체 cycle {fmt(r.cycle_seconds,3)}초</small></div></div>
      <div className="moe-health-strip"><span>실제 계산 <b>{live?str(compute.inference_device):'정지'}</b> · {str(compute.gpu_name,'GPU 정보 없음')}</span><span>VRAM <b>{bytes(live?compute.allocated_bytes:0)}</b> · RAM <b>{bytes(live?r.worker_ram_bytes:0)}</b></span><span>모델 <b>{fmt(num(r.parameters)/1e9,4)}B</b> · 파일 {bytes(r.checkpoint_bytes)}</span><span>optimizer <b>{fmt(r.optimizer_updates,0)}회</b></span></div>
    </section>
    <section className="moe-flow-region"><header><div><small>MODEL OUTPUT TO LEARNING</small><h2>판단부터 학습까지</h2></div><span>최근 상태와 값</span></header><StageTrack steps={[
      {label:'시장 Expert',detail:fmt(r.expert_count,0)+'개 · '+(stages.market?'실행':'대기'),state:stages.market?'done':'waiting'},
      {label:'시장 상태 · routing/fusion',detail:stages.state?'공통 상태 구성':'전문가 출력 대기',state:stages.state?'done':'waiting'},
      {label:'Policy · Controller',detail:stages.policy||stages.controller?'정책 근거와 최종 결정':'입력 대기',state:stages.controller?'done':stages.policy?'active':'waiting'},
      {label:'Action · target weight',detail:str(decision.action,'대기')+' · '+pct(decision.target_weight),state:stages.action?'done':'waiting'},
      {label:'Paper execution',detail:list(r.fills).length+' 최근 체결',state:list(r.fills).length?'done':'waiting'},
      {label:'Reward · Replay',detail:'reward '+fmt(learning.reward_points??r.reward_points,5)+' · eligible '+fmt(replay.eligible,0),state:num(replay.eligible)?'active':'waiting'},
      {label:'Optimizer',detail:fmt(r.optimizer_updates,0)+'회 · loss '+fmt(learning.loss,6),state:num(r.optimizer_updates)?'done':'waiting'},
    ]}/></section>
    <div className="moe-ledger-layout"><section><header className="region-heading"><div><small>PAPER ACCOUNT</small><h2>계좌 변화</h2></div><span>현재 포지션: {fmt(Object.values(obj(r.books)).reduce<number>((n,b)=>n+list(obj(b).positions).length,0),0)}종목</span></header><DataTable headers={['통화','NAV','현금','누적 손익 · 수익률','체결','총 비용']} rows={books}/><JsonDetails title="포지션 상세" data={r.books}/></section><section className="moe-learning-summary"><header><div><small>ONLINE LEARNING</small><h2>최근 업데이트</h2></div></header><DataTable headers={['값','최근 기록']} rows={[
      ['reward',fmt(learning.reward_points??r.reward_points,6)],['loss',fmt(learning.loss,6)],['samples',fmt(learning.samples,0)+'건'],['학습 소요',fmt(learning.seconds,3)+'초'],['Replay 전체 / 학습 가능',fmt(replay.total,0)+' / '+fmt(replay.eligible,0)],['손익 결과 대기',fmt(replay.pending,0)+'건'],['최근 update',time(learning.updated_at)]]}/></section></div>
    <section className="moe-flow-region"><header><div><small>PAPER FILLS</small><h2>최근 가상 체결</h2></div><span>{list(r.fills).length}건</span></header><FillHistory fills={r.fills} eth/></section>
  </>;
}
