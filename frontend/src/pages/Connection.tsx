import { useEffect, useState } from 'react';
import { api, endpoints } from '../api';
import { usePolling } from '../hooks';
import { obj, str, time } from '../data';
import { ActionButton, ErrorState, JsonDetails, Loading, SectionLabel, Signal } from '../components/ui';

export function Connection() {
  const q = usePolling('/api/provider'), p = q.data;
  const [environment, setEnvironment] = useState('real'), [account, setAccount] = useState(''), [appKey, setAppKey] = useState(''), [secret, setSecret] = useState(''), [ip, setIp] = useState(''), [edited, setEdited] = useState(false);
  useEffect(() => { if (p?.environment && !edited) setEnvironment(p.environment); }, [p?.environment, edited]);
  if (!p) return q.error ? <ErrorState message={q.error} retry={q.refresh}/> : <Loading/>;
  const test = obj(p.last_test);
  return <>
    {q.error && <ErrorState message={'마지막 연결 상태 표시 중 · '+q.error} retry={q.refresh}/>}
    <section className="connection-panel">
      <header className="connection-status"><SectionLabel eyebrow="PROVIDER CONNECTION" title="시세 제공자 연결" detail="인증 설정은 모델·계좌 실행 제어와 분리됩니다."/><Signal label="저장된 인증" value={p.saved?'등록됨':'미등록'} tone={p.saved?'good':'warn'}/></header>
      <div className="connection-summary"><div><small>마지막 인증 확인</small><strong>{test.ok?'성공':'확인 필요'}</strong><span>{time(test.checked_at||test.timestamp)}</span></div><div><small>접속 환경</small><strong>{str(p.environment)}</strong><span>실제 주문 설정에는 영향 없음</span></div><div><small>공인 IP</small><strong>{ip||'조회 전'}</strong><span>제공자 등록용</span></div></div>
      <form className="credential-form" onSubmit={(e)=>e.preventDefault()}>
        <label>접속 환경<select value={environment} onChange={(e)=>{setEdited(true);setEnvironment(e.target.value);}}><option value="real">실시간 시세</option><option value="paper">모의 시세</option></select></label>
        <label>계좌 번호<input value={account} onChange={(e)=>setAccount(e.target.value)} autoComplete="off"/></label>
        <label>App Key<input type="password" value={appKey} onChange={(e)=>setAppKey(e.target.value)} autoComplete="new-password"/></label>
        <label>Secret<input type="password" value={secret} onChange={(e)=>setSecret(e.target.value)} autoComplete="new-password"/></label>
        <div className="credential-actions"><ActionButton label="인증 정보 저장 · 연결" disabled={!appKey||!secret} task={async()=>{const result=await api.command(endpoints.connect,{environment,account,app_key:appKey,secret});setAppKey('');setSecret('');setAccount('');setEdited(false);return result;}}/><ActionButton label="저장된 인증으로 재확인" path={endpoints.testProvider} body={{provider:'kiwoom',environment}} disabled={!p.saved}/><ActionButton label="공인 IP 조회" task={async()=>{const r=await api.publicIp();setIp(str(r.ip||r.public_ip));return r;}}/></div>
      </form>
      <JsonDetails title="연결 진단 정보" data={p}/>
    </section>
  </>;
}
