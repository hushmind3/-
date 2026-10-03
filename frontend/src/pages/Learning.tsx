import { at, bytes, fmt, num, obj, pct, time } from '../data';
import type { StatusResponse } from '../types';
import { LearningPair } from '../components/operations';
import { DataTable, JsonDetails, SectionLabel, Signal, StageTrack } from '../components/ui';

export function Learning({status:s}:{status:StatusResponse}) {
  const l=obj(s.learning), m=obj(s.metrics), champion=s.model_runtime?.champion||{}, replay=obj(champion.replay),
    pending=num(replay.pending??l.replay_pending_count), eligible=num(replay.eligible??l.eligible_backlog),
    untrained=num(replay.untrained??l.replay_untrained_count), quarantined=num(replay.quarantined??l.replay_quarantined_count),
    unsupported=num(replay.unsupported??l.replay_unsupported_count), updates=num(champion.optimizer_updates)+num(s.model_runtime?.candidate?.optimizer_updates),
    done=Array.isArray(replay.daily)?replay.daily.reduce<number>((sum,row)=>sum+num(obj(row).completed),0):undefined;
  return <>
    <section className="learning-map"><header><div><small>EXPERIENCE → WEIGHT UPDATE</small><h2>경험 처리 진행</h2></div><Signal label="누적 optimizer" value={fmt(updates,0)+'회'} tone={updates?'good':'neutral'}/></header>
      <StageTrack steps={[
        {label:'행동 경험 생성',detail:'가상 판단·체결·계좌 변화 저장',state:updates?'done':'waiting'},
        {label:'reward 결과 대기',detail:fmt(pending,0)+'건 · 후속 시세 필요',state:pending?'active':'done'},
        {label:'replay 저장 · 미학습',detail:fmt(untrained,0)+'건',state:untrained?'active':'waiting'},
        {label:'학습 가능',detail:fmt(eligible,0)+'건 · reward 확정',state:eligible?'active':'waiting'},
        {label:'optimizer 업데이트',detail:'Champion '+fmt(champion.optimizer_updates,0)+'회 · Candidate '+fmt(s.model_runtime?.candidate?.optimizer_updates,0)+'회',state:updates?'done':'waiting'},
      ]}/>
      <div className="backlog-strip"><div><small>손익 확인 대기</small><strong>{fmt(pending,0)}</strong><span>미성숙 reward</span></div><div><small>학습 가능</small><strong>{fmt(eligible,0)}</strong><span>확정 경험</span></div><div><small>미학습</small><strong>{fmt(untrained,0)}</strong><span>아직 optimizer 미투입</span></div><div><small>학습 보류</small><strong>{fmt(quarantined+unsupported,0)}</strong><span>보류 {fmt(quarantined,0)} · 호환 불가 {fmt(unsupported,0)}</span></div><div><small>Replay DB</small><strong>{bytes(replay.bytes??l.replay_bytes)}</strong><span>전체 {fmt(replay.total??l.replay_current,0)}건</span></div><div><small>완료 경험</small><strong>{fmt(done,0)}</strong><span>누계</span></div></div>
    </section>
    <section className="learning-models"><SectionLabel eyebrow="MODEL COMPARISON" title="Champion · Candidate 학습 비교" detail={s.learning_enabled?'학습 허용':'학습 중지'}/><LearningPair status={s}/></section>
    <section className="learning-recent"><SectionLabel eyebrow="RECENT UPDATES" title="최근 학습 결과"/><DataTable headers={['모델','상태','samples','소요 시간','loss','reward','최근 업데이트']} rows={(['champion','candidate'] as const).map((role)=>{const run=s.model_runtime?.[role]||{},detail=obj(run.learning),last=obj(m[role+'_last_completed_round']);return[role==='champion'?'Champion':'Candidate',run.learning_active?'학습 중':run.loaded&&s.learning_enabled?'경험 대기':'중지',fmt(detail.samples??last.samples,0)+'건',fmt(detail.seconds??last.seconds,3)+'초',fmt(detail.loss??last.loss,6),fmt(detail.reward_points??last.reward_points,6),time(detail.updated_at||last.updated_at||last.time)];})}/></section>
    <section className="coverage-section"><SectionLabel eyebrow="INPUT COVERAGE" title="시간 간격별 입력 확보" detail="완전 충족 종목과 실제 관측 종목을 구분"/><DataTable headers={['시간 간격','기록 확보율','입력 보유 / 관측','필요 기록 충족','Champion samples','Candidate samples']} rows={Object.entries(obj(l.multiscale_input_status)).map(([key,value])=>{const row=obj(value);return[key,pct(row.mean_history_coverage),fmt(row.available_symbols,0)+' / '+fmt(row.observed_symbols,0),fmt(row.complete_history_symbols,0),fmt(at(m,'champion_last_completed_round.timeframe_samples.'+key),0),fmt(at(m,'candidate_last_completed_round.timeframe_samples.'+key),0)];})}/><p className="subtle">기록 충족은 필요한 입력 길이를 채운 종목 수입니다. 일부 기록도 학습 입력으로 쓰일 수 있습니다.</p></section>
    <div className="diagnostic-region"><JsonDetails title="Reward 산정 기준" data={l.reward_credit}/><JsonDetails title="날짜별 생성·학습 완료" data={l.daily_learning||replay.daily}/><JsonDetails title="학습 보류 사유" data={{blocked:l.blocked_replay,reasons:replay.blocked_reasons}}/></div>
  </>;
}
