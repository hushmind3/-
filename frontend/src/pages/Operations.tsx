import { useMemo, useState } from 'react';
import { usePolling } from '../hooks';
import { at, fmt, list, num, obj, pct, rows, state, str, time } from '../data';
import type { Data, StatusResponse } from '../types';
import { AccountsComparison, ExecutionModes, FillHistory, ModelComparison, RuntimeCommands } from '../components/operations';
import { DataTable, Disclosure, JsonDetails, Signal } from '../components/ui';

export function Operations({ status:s }: { status:StatusResponse }) {
  const assembly=usePolling('/api/assembly/status',3000), moe=usePolling('/api/trading-moe/status',3000),
    trial=obj(assembly.data?.worker), candidateOccupied=!!trial.alive,
    models=s.model_runtime||{}, gpu=obj(s.physical_gpu), c=models.champion||{}, d=models.candidate||{};
  const nodes=[
    {kind:'feed',name:'시장 Feed',value:s.feed_running?'수신 중':'정지',desc:'공유 시세 입력',tone:s.feed_running?'good':'neutral'},
    {kind:'champion',name:'Champion',value:state(c.status),desc:c.loaded?'모델 적재 · '+str(c.compute_device||c.device):'모델 미적재',tone:c.error?'bad':c.loaded?'good':'neutral'},
    {kind:'candidate',name:candidateOccupied?'Candidate · 시험 중':'Candidate',value:candidateOccupied?state(trial.evaluation_stage||assembly.data?.candidate?.evaluation_state):state(d.status),desc:candidateOccupied?'시험용 paper ledger 별도':'운영 모델 worker',tone:candidateOccupied?'warn':d.loaded?'good':'neutral'},
    {kind:'moe',name:'독립 TradingMoE',value:moe.data?.gpu_waiting?'GPU 대기':state(moe.data?.status),desc:'별도 worker · 계좌 · 정지 명령',tone:moe.data?.alive?'good':'neutral'},
    {kind:'gpu',name:str(s.gpu,'GPU'),value:gpu.memory_total_mb?fmt(gpu.utilization_percent,0)+'%':'—',desc:'VRAM '+fmt(num(gpu.memory_used_mb)/1024)+' / '+fmt(num(gpu.memory_total_mb)/1024)+' GiB',tone:'neutral'},
  ];
  return <>
    <section className="runboard"><header><div><small>LIVE SYSTEM MAP</small><h2>누가 실행 중인지</h2></div><span>각 항목은 서로 다른 실행·저장·정지 단위입니다.</span></header><div className="runtime-nodes">{nodes.map((n)=><article key={n.kind} className={'runtime-node '+n.kind}><Signal label={n.name} value={n.value} tone={n.tone as 'good'|'warn'|'bad'|'neutral'}/><small>{n.desc}</small>{n.kind==='moe'&&<a href="#trading-moe">독립 화면 ↗</a>}{n.kind==='candidate'&&candidateOccupied&&<a href="#promotionTrial">승급전 ↗</a>}</article>)}</div><div className="runtime-relations"><b>의존 관계</b><span>Feed → Champion / Candidate</span><span>Assembly 시험은 Candidate worker slot 사용</span><span>독립 TradingMoE는 공유 GPU만 경쟁, 정지는 별도</span></div></section>
    <div className="control-strip"><ExecutionModes status={s}/><RuntimeCommands status={s}/></div>
    <ModelComparison status={s}/>
    {assembly.data?.candidate&&<section className="trial-peek"><header><div><small>CURRENT COMPETITION</small><h2>진행 중인 승급 시험</h2></div><a href="#promotionTrial">시험 결과 보기 ↗</a></header><div className="trial-peek-data"><strong>{str(assembly.data.candidate.candidate_id)}</strong><span>{state(assembly.data.candidate.evaluation_state)}</span><span>{str(assembly.data.candidate.reason,'같은 조건 평가 진행 중')}</span><b>Replay {pct(at(assembly.data,'candidate.scores.replay.delta'))} · Paper {pct(at(assembly.data,'candidate.scores.paper.delta'))}</b></div></section>}
    <AccountsComparison status={s}/>
    <Disclosure title="최근 주문·체결 기록"><FillHistory fills={at(s,'paper_account.fills')}/></Disclosure>
    <JsonDetails title="진단 · 최근 오류와 계좌 출처" data={{errors:{champion:c.error,candidate:d.error,assembly:assembly.error,trading_moe:moe.data?.error},accounts:at(s,'account_observability')}}/>
  </>;
}

export function Markets({status:s}:{status:StatusResponse}) {
  const [search,setSearch]=useState(''),[fresh,setFresh]=useState(false),[market,setMarket]=useState('all'),[action,setAction]=useState('all'),[sort,setSort]=useState('symbol'),[selected,setSelected]=useState<string[]>([]), marketRows=rows(s.markets);
  const instruments=useMemo(()=>rows(s.instruments).filter((i)=>(!fresh||!!i.fresh)&&(action==='all'||str(at(obj(i.decision),'action'),'')===action)&&(market==='all'||list(marketRows.find((m)=>m.key===market)?.symbols).includes(str(i.symbol,'')))&&[i.symbol,i.name,i.market].some((v)=>str(v,'').toLowerCase().includes(search.toLowerCase()))).sort((a,b)=>sort==='price'?num(obj(b.quote).close)-num(obj(a.quote).close):sort==='volume'?num(obj(b.quote).volume)-num(obj(a.quote).volume):sort==='action'?str(obj(a.decision).action).localeCompare(str(obj(b.decision).action)):str(a.symbol).localeCompare(str(b.symbol),'ko')),[s.instruments,marketRows,search,fresh,market,action,sort]);
  const selectedRows=selected.map((symbol)=>rows(s.instruments).find((item)=>item.symbol===symbol)).filter((item):item is Data=>!!item);
  return <>
    <section className="market-workspace"><header className="market-workspace-head"><div><small>ONE MARKET WORKSPACE</small><h2>시장을 고르고 종목을 스캔</h2></div><Signal label="현재 표시" value={fmt(instruments.length,0)+'종목'} tone="neutral"/></header>
      <nav className="market-picker" aria-label="시장 필터"><button className={market==='all'?'chosen':''} onClick={()=>setMarket('all')}>전체 <b>{fmt(marketRows.reduce((n,row)=>n+num(row.count??list(row.symbols).length),0),0)}</b></button>{marketRows.map((m)=><button key={str(m.key)} className={market===m.key?'chosen':''} onClick={()=>setMarket(str(m.key))}>{str(m.label||m.key)} <b>{fmt(m.fresh_count,0)}/{fmt(m.count??list(m.symbols).length,0)}</b></button>)}</nav>
      <div className="market-filters"><label className="search-field">종목 검색<input value={search} onChange={(e)=>setSearch(e.target.value)} placeholder="종목 코드 또는 이름"/></label><label>행동<select value={action} onChange={(e)=>setAction(e.target.value)}><option value="all">전체</option><option value="BUY">BUY</option><option value="HOLD">HOLD</option><option value="SELL">SELL</option></select></label><label>정렬<select value={sort} onChange={(e)=>setSort(e.target.value)}><option value="symbol">종목명</option><option value="price">가격</option><option value="volume">거래량</option><option value="action">행동</option></select></label><label className="check-field"><input type="checkbox" checked={fresh} onChange={(e)=>setFresh(e.target.checked)}/>최근 5분 수신만</label><span>{instruments.length}개 검색 결과 · 최대 4개 비교</span></div>
      {selectedRows.length>0&&<section className="selection-shelf"><header><b>선택 종목 비교</b><span>{selectedRows.length}/4</span><button onClick={()=>setSelected([])}>선택 지우기</button></header><div className="selected-instruments">{selectedRows.map((i)=><article key={str(i.symbol)}><b>{str(i.symbol)}</b><strong>{fmt(at(i,'quote.close'),4)}</strong><span>{str(at(i,'decision.action'))} · 목표 {pct(at(i,'decision.target_weight'))}</span><small>거래량 {fmt(at(i,'quote.volume'),0)}</small></article>)}</div></section>}
      <DataTable headers={['선택','종목','이름','시장','시세 시각','현재가','거래량','판단','목표 비중']} rows={instruments.map((i)=>{const q=obj(i.quote),d=obj(i.decision),symbol=str(i.symbol);return[<input aria-label={symbol+' 비교에 추가'} type="checkbox" checked={selected.includes(symbol)} disabled={!selected.includes(symbol)&&selected.length>=4} onChange={(e)=>setSelected((items)=>e.target.checked?[...items,symbol]:items.filter((x)=>x!==symbol))}/>,<strong className="mono">{symbol}</strong>,str(i.name),str(i.market),time(q.date),<span className="numeric">{fmt(q.close,4)}</span>,<span className="numeric">{fmt(q.volume,0)}</span>,<b className={'action-word '+str(d.action,'')}>{str(d.action)}</b>,pct(d.target_weight)];})}/>
      <Disclosure title="시세 입력 충족률 · universe 상세"><JsonDetails title="현재 입력 확보 현황" data={s.input_availability}/><JsonDetails title="추가 universe 설정" data={s.universe_expansion}/></Disclosure>
    </section>
  </>;
}
