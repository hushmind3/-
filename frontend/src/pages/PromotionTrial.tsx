import { usePolling } from '../hooks';
import { fmt, num, obj, pct, state, str, time } from '../data';
import { ActionButton, DataTable, ErrorState, Loading, SectionLabel, Signal, StageTrack } from '../components/ui';
import { api } from '../api';
import type { Data } from '../types';

export function PromotionTrial(){
 const q=usePolling('/api/assembly/status',3000),a=q.data;
 if(!a)return q.error?<ErrorState message={q.error} retry={q.refresh}/>:<Loading/>;
 const c=a.candidate||{},ch=a.champion||{},w=obj(a.worker),scores=obj(c.scores),replay=obj(scores.replay),paper=obj(scores.paper),stage=str(c.evaluation_state,'waiting');
 const phase=stage.includes('replay')?'replay':stage.includes('paper')?'paper':paper.candidate?'decision':replay.candidate?'paper':'replay';
 return <>
  {q.error&&<ErrorState message={q.error} retry={q.refresh}/>}
  <section className="trial-hero"><header><SectionLabel eyebrow="PROMOTION EVALUATION" title="Candidate 승급 시험" detail="Champion과 동일한 입력·비용 조건의 결과를 나란히 비교합니다."/><Signal label="현재 시험" value={state(stage)} tone={w.alive?'good':'neutral'}/></header><div className="trial-identities"><div><small>현재 기준 모델</small><strong>Champion</strong><span>{str(ch.candidate_id)}</span></div><div className="versus">VS</div><div><small>시험 대상</small><strong>Candidate</strong><span>{str(c.candidate_id)}</span></div></div><p className="trial-reason">{str(c.mutation_description,'시험 Candidate 없음')} · {str(c.reason,typeof w.message==='string'?w.message:'시험 진행 상태를 확인 중')}</p><div className="action-row"><ActionButton label="현재 Candidate 시험 시작" path={api.assembly('trial/start')} disabled={!a.candidate||!!w.alive}/><ActionButton label="현재 Candidate 시험 정지" path={api.assembly('trial/stop')} disabled={!w.alive}/><ActionButton label="현재 Candidate 탈락 · 다음 후보" path={api.assembly('next')} tone="warn"/></div></section>
  <section className="trial-stage"><SectionLabel eyebrow="EVALUATION GATES" title="시험 진행 단계" detail={fmt(w.bars_current,0)+' / '+fmt(w.bars_required,0)+' 시점'}/><StageTrack steps={[{label:'Replay 예선',detail:'과거 구간의 빠른 비용 포함 비교',state:replay.candidate?'done':phase==='replay'?'active':'waiting'},{label:'Paper shadow 평가',detail:'실행·체결 조건 비교',state:paper.candidate?'done':phase==='paper'?'active':'waiting'},{label:'동일 조건 확인',detail:'시장 입력·비용·기간 확인',state:w.comparison_valid?'done':w.alive?'active':'waiting'},{label:'승급 판정',detail:state(stage),state:stage==='promoted'?'done':stage==='rejected'?'blocked':'waiting'}]}/></section>
  <section className="trial-head-to-head"><SectionLabel eyebrow="SAME MARKET · SAME COST" title="Champion 대비 결과" detail="차이가 양수면 Candidate가 우세한 지표입니다."/><div className="trial-accounts">{([['Replay 예선',replay],['Paper shadow',paper]] as [string,Data][]).map(([title,value])=><article className="trial-gate" key={title}><h3>{title}</h3><DataTable headers={['지표','Candidate','Champion','차이']} rows={['net_return','max_drawdown','trades','fees','seconds'].map(key=>[{net_return:'비용 차감 수익률',max_drawdown:'최대 손실폭',trades:'체결 수',fees:'거래 비용',seconds:'평가 시간'}[key],key==='net_return'||key==='max_drawdown'?pct(obj(value.candidate)[key]):fmt(obj(value.candidate)[key],key==='seconds'?3:0),key==='net_return'||key==='max_drawdown'?pct(obj(value.champion)[key]):fmt(obj(value.champion)[key],key==='seconds'?3:0),<b className={num(value.delta)<0?'negative':'positive'}>{key==='net_return'||key==='max_drawdown'?pct(value.delta):'—'}</b>])}/></article>)}</div></section>
  <footer className="trial-footer"><span>판정 근거 · {str(c.reason,'아직 확정된 판정이 없습니다.')}</span><span>Candidate 생성 {time(c.created_at)} · 부모 {str(c.parent_id)}</span><a href="#assembly">자동 조립·후보 이력 보기 →</a></footer>
 </>;
}
