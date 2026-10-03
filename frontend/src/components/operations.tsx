import { api, endpoints } from '../api';
import { at, fmt, num, obj, pct, rows, str, time } from '../data';
import type { Data, Json, StatusResponse } from '../types';
import { ActionButton, DataTable, Disclosure, Signal, Toggle } from './ui';

export function ExecutionModes({ status: s }: { status: StatusResponse }) {
  return <section className="control-block mode-controls"><header><div><small>INDEPENDENT FLAGS</small><h2>실행 모드</h2></div><span>Feed가 켜져 있어도 각 모드는 독립적으로 전환됩니다.</span></header><div className="switch-row"><Toggle label="모델 판단" enabled={!!s.observe_enabled} path={endpoints.modes} field="observe_enabled"/><Toggle label="가상 체결" enabled={!!s.paper_enabled} path={endpoints.modes} field="paper_enabled"/><Toggle label="경험 학습" enabled={!!s.learning_enabled} path={endpoints.modes} field="learning_enabled"/></div><div className="mode-definitions"><span><b>판단</b> 새 시세로 행동 결정</span><span><b>체결</b> paper 계좌 반영</span><span><b>학습</b> 확정 경험으로 가중치 업데이트</span></div></section>;
}

export function RuntimeCommands({ status: s }: { status: StatusResponse }) {
  return <section className="control-block command-block"><header><div><small>SHARED FEED CONTROL</small><h2>시세 수집 제어</h2></div><Signal label="실제 주문" value={s.real_orders_enabled ? '허용' : '차단'} tone={s.real_orders_enabled ? 'bad' : 'good'}/></header><div className="action-row"><ActionButton label={s.feed_running ? 'Feed 수신 중' : '시장 Feed 시작'} disabled={!!s.feed_running} path={endpoints.start} body={{mode:'live'}}/><ActionButton label="시세 재연결" path={endpoints.reconnect}/><ActionButton label="웹서버만 재시작" path={endpoints.serverRestart}/><ActionButton label="전체 정지" tone="bad" path={endpoints.stop}/><ActionButton label="두 장기 운영계좌 초기화" tone="bad" path={endpoints.resetAccounts} confirm="Champion과 Candidate의 장기 가상계좌만 초기화합니다. 가중치와 replay는 유지합니다. 계속할까요?"/></div><p>시작은 Feed만 켭니다. 전체 정지는 Feed·운영 모델·Assembly 시험을 저장 후 정지합니다. 별도 TradingMoE worker는 자동매매 화면에서 정지합니다.</p></section>;
}

function runtime(scope: Data, field: string) { return at(scope, field); }
export function ModelComparison({ status: s }: { status: StatusResponse }) {
  const decisions = rows(s.decisions);
  const roles = ['champion','candidate'] as const;
  return <section className="model-comparison"><header className="region-heading"><div><small>MODEL WORKERS · SAME FIELDS</small><h2>Champion ↔ Candidate</h2></div><span>worker, 계좌, replay는 각자 독립</span></header><div className="model-actions">{roles.map((role) => {const r=s.model_runtime?.[role]||{}, name=role==='champion'?'Champion':'Candidate'; return <div key={role} className={'model-control '+role}><strong>{name}</strong><Signal label="상태" value={str(r.status,'정지')} tone={r.error?'bad':r.loaded?'good':'neutral'}/><div className="action-row"><ActionButton label={name+' 시작'} path={api.model(role,'start')} disabled={r.status==='saving'||((!!r.loaded||!!r.requested)&&r.status!=='error')}/><ActionButton label={name+' 저장 후 정지'} path={api.model(role,'stop')} disabled={r.status==='saving'||(!r.loaded&&!r.requested)}/></div>{r.error&&<p className="error-text">{str(r.error)}</p>}</div>;})}</div><DataTable headers={['비교 항목','Champion','Candidate']} rows={[
    ['최근 판단', decisionText(decisions,'champion'), decisionText(decisions,'candidate')],
    ['적재 · 장치', modelDevice(s.model_runtime?.champion), modelDevice(s.model_runtime?.candidate)],
    ['최근 시세 시점', time(s.model_runtime?.champion?.last_decision), time(s.model_runtime?.candidate?.last_decision)],
    ['판단 소요', seconds(s.model_runtime?.champion?.decision_seconds), seconds(s.model_runtime?.candidate?.decision_seconds)],
    ['학습 · 누적 업데이트', training(s,'champion'), training(s,'candidate')],
    ['Replay 학습 가능 / 손익 대기', replay(s.model_runtime?.champion), replay(s.model_runtime?.candidate)],
  ]}/><p className="subtle">{str(s.model_runtime?.candidate?.source).includes('조립 Candidate 시험')?'Candidate 실행 슬롯에서 Assembly 시험 중입니다. 시험 계좌는 아래 별도 영역에서 확인하세요.':'일반 Candidate worker가 운영 계좌로 실행 중입니다.'}</p></section>;
}
function decisionText(decisions: Data[], role: string) {const d=decisions.find((x)=>x.role===role); return d?str(d.symbol)+' · '+str(d.action)+' · 목표 '+pct(d.target_weight):'새 판단 대기';}
function modelDevice(r: Data|undefined) {return str(r?.loaded?'적재됨':'미적재')+' · '+str(r?.compute_device||r?.device,'—');}
function seconds(value: Json|undefined) {return value==null?'—':fmt(value,3)+'초';}
function training(s: StatusResponse, role: 'champion'|'candidate') {const r=s.model_runtime?.[role]; return (r?.learning_active?'학습 중':'학습 대기')+' · '+fmt(r?.optimizer_updates,0)+'회';}
function replay(r: Data|undefined) {return fmt(runtime(r||{},'replay.eligible')??runtime(r||{},'replay.remaining_for_update'),0)+'건 · '+fmt(runtime(r||{},'replay.pending'),0)+'건 결과 대기';}

type AccountLine={owner:string; currency:string; book:Data; test:boolean};
function accountRows(s: StatusResponse): AccountLine[] {
  const rowsOut:AccountLine[]=[];
  for(const role of ['champion','candidate'] as const){const model=s.model_runtime?.[role]||{}, owner=role==='champion'?'Champion':'Candidate', test=str(model.source).includes('조립 Candidate 시험')||model.account_scope==='trial'; const books=test?model.books:(model.books||(role==='champion'?s.paper_financials:at(s,'candidate_live_account.books'))); for(const [currency,book] of Object.entries(obj(books))) rowsOut.push({owner,currency,book:obj(book),test});}
  return rowsOut;
}
function held(book:Data):Data[]{const p=book.positions; return Array.isArray(p)?p.filter((value):value is Data=>!!value&&typeof value==='object'&&!Array.isArray(value)): Object.entries(obj(p)).map(([symbol,value])=>({symbol,...obj(value),mark:obj(book.marks)[symbol]}));}
export function AccountsComparison({status:s}:{status:StatusResponse}) {
  const entries=accountRows(s), operating=entries.filter((e)=>!e.test), trials=entries.filter((e)=>e.test);
  const table=(list:AccountLine[])=> <DataTable headers={['계좌','순자산 / 손익','현금','포지션','체결','총 비용','보유 종목']} rows={list.map(({owner,currency,book})=>{const pnl=num(book.net_pnl??(num(book.equity)-num(book.initial_cash))), pos=held(book);return[owner+' · '+currency, <strong className={pnl<0?'negative':'positive'}>{fmt(book.equity)} <small>{currency}</small><small> 손익 {fmt(pnl)} · {pct(book.net_return_rate??(num(book.initial_cash)?pnl/num(book.initial_cash):0))}</small></strong>,fmt(book.cash),pos.length+'종목',fmt(book.trade_count,0)+'회',fmt(book.costs??(num(book.fees)+num(book.slippage)+num(book.spread)+num(book.sell_tax))),pos.length?pos.map((p)=>str(p.symbol)).join(', '):'현금 보유'];})}/>;
  return <section className="account-comparison"><header className="region-heading"><div><small>PAPER LEDGERS</small><h2>가상계좌 비교</h2></div><span>장기 운영과 시험 계좌를 분리 표시</span></header>{table(operating)}{trials.length>0&&<div className="trial-ledger"><h3>Assembly 시험계좌 <span>장기 운영계좌와 별도</span></h3>{table(trials)}</div>}{entries.map((e)=>{const positions=held(e.book);return positions.length?<Disclosure key={e.owner+e.currency} title={e.owner+' · '+e.currency+' 보유 종목 상세'}><DataTable headers={['종목','수량','평단','현재가','평가손익','비중']} rows={positions.map((p)=>[str(p.symbol),fmt(p.quantity,6),fmt(p.average_cost,4),fmt(p.mark??p.current_price??p.price,4),fmt(p.unrealized_pnl),pct(p.weight)])}/></Disclosure>:null;})}</section>;
}

export function FillHistory({ fills, eth=false }: { fills: Json|undefined; eth?: boolean }) {
  const records=rows(fills).slice(-30).reverse();
  return <DataTable headers={['시각','종목','매매','수량','체결가','수수료','실현손익']} empty="아직 체결이 없습니다." rows={records.map((f)=>[time(f.timestamp||f.date||f.time),str(f.symbol),str(f.side||f.action),fmt(num(f.quantity)*(eth?.001:1),6),fmt(num(f.price||f.fill_price)*(eth?1000:1),4),fmt(f.fee||f.fees,4),fmt(f.realized_pnl||f.net_pnl,4)])}/>;
}

export function LearningPair({status:s}:{status:StatusResponse}) {
  const models=s.model_runtime||{};
  return <DataTable headers={['모델','실행 상태','optimizer 누계','최근 samples · 시간','loss · reward','미학습 replay · 결과 대기']} rows={(['champion','candidate'] as const).map((role)=>{const m=models[role]||{}, l=obj(m.learning),r=obj(m.replay);return[role==='champion'?'Champion':'Candidate',m.learning_active?'학습 중':s.learning_enabled&&m.loaded?'경험 대기':'중지',fmt(m.optimizer_updates,0)+'회',fmt(l.samples??at(s.metrics,role+'_last_completed_round.samples'),0)+'건 · '+fmt(l.seconds??s.learning?.candidate_update_seconds,3)+'초','loss '+fmt(l.loss,6)+' · reward '+fmt(l.reward_points,6),fmt(r.eligible??r.remaining_for_update,0)+'건 · '+fmt(r.pending,0)+'건'];})}/>;
}
