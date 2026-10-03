import { useMemo, useState } from 'react';
import { usePolling } from '../hooks';
import { bytes, fmt, obj, str, time } from '../data';
import type { StatusResponse } from '../types';
import { DataTable, Disclosure, ErrorState, JsonDetails, SectionLabel, Signal } from '../components/ui';

export function Details({ status: s }: { status: StatusResponse }) {
  const r=usePolling('/api/runtime'), [query,setQuery]=useState(''), runtime=obj(r.data), gpu=obj(s.physical_gpu), logs=useMemo(()=>{
    const source=typeof s.logs==='string'?s.logs.split('\n'):(Array.isArray(s.logs)?s.logs.map((line)=>str(line)):[]);
    return source.filter((line)=>line.toLowerCase().includes(query.toLowerCase())).slice(-300).reverse();
  },[s.logs,query]);
  const models=s.model_runtime||{};
  return <>
    {r.error&&<ErrorState message={r.error}/>}
    <section className="observability-overview"><SectionLabel eyebrow="RUNTIME OBSERVABILITY" title="프로세스와 자원" detail="운영 판단은 각 메뉴에서, 이 화면은 원인 추적에 사용합니다."/><div className="resource-status"><Signal label="웹서버" value={s.running?'실행 중':'정지'} tone={s.running?'good':'bad'}/><Signal label="Feed" value={s.feed_running?'수신 중':'정지'} tone={s.feed_running?'good':'neutral'}/><Signal label="GPU 연산" value={fmt(gpu.utilization_percent,0)+'%'} tone="neutral"/><Signal label="GPU 메모리" value={fmt(gpu.memory_used_mb,0)+' / '+fmt(gpu.memory_total_mb,0)+' MB'} tone="neutral"/><Signal label="시스템 RAM" value={bytes(runtime.ram_used_bytes)} tone="neutral"/><Signal label="Candidate" value={str(models.candidate?.status)} tone={models.candidate?.error?'bad':'neutral'}/></div></section>
    <section className="log-explorer"><SectionLabel eyebrow="RECENT EVENTS" title="서버 로그" detail={logs.length+'줄 표시 · 최근 300줄 범위'}/><label className="search-field">로그 검색<input value={query} onChange={(e)=>setQuery(e.target.value)} placeholder="오류 문구, 프로세스, 종목"/></label><div className="log-list" role="log">{logs.length?logs.map((line,i)=><pre className="log-row" key={i}>{line}</pre>):<p className="empty-state">검색 결과가 없습니다.</p>}</div></section>
    <section className="diagnostic-grid"><Disclosure title="모델 worker 상세"><DataTable headers={['모델','상태','장치','RAM','VRAM','최근 판단','최근 오류']} rows={(['champion','candidate'] as const).map((key)=>{const m=models[key]||{};return[key==='champion'?'Champion':'Candidate',str(m.status),str(m.compute_device||m.device),bytes(m.ram_weight_bytes),bytes(m.gpu_weight_bytes),time(m.last_decision),str(m.error,'없음')];})}/></Disclosure><Disclosure title="시장 수집 상태"><JsonDetails title="Feed 지표" data={s.feed_metrics}/><JsonDetails title="입력 확보 현황" data={s.input_availability}/></Disclosure><Disclosure title="paper·행동 진단"><JsonDetails title="계좌 관측" data={s.account_observability}/><JsonDetails title="출력 분포" data={s.output_diagnostics}/></Disclosure><Disclosure title="프로세스 runtime 원본"><JsonDetails title="Runtime 원본 데이터" data={r.data}/><JsonDetails title="Backtest 설정" data={s.backtest}/></Disclosure></section>
  </>;
}
