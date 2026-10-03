import { useEffect } from 'react';
import { CommandProvider, ErrorState, Loading, Signal } from './components/ui';
import { pageFromHash, pages, time, type Page } from './data';
import { useHash, usePolling } from './hooks';
import { Operations, Markets } from './pages/Operations';
import { Learning } from './pages/Learning';
import { PromotionTrial } from './pages/PromotionTrial';
import { Connection } from './pages/Connection';
import { Details } from './pages/Details';
import { Experts } from './pages/Experts';
import { TradingMoE } from './pages/TradingMoE';
import { Assembly } from './pages/Assembly';

const descriptions: Record<Page, string> = {
  control: 'Feed와 두 운영 모델, 가상계좌를 한 화면에서 관리합니다.',
  markets: '시장 상태에서 종목별 최근 판단까지 이어서 훑습니다.',
  learning: '경험이 보상 확정과 replay를 거쳐 가중치 업데이트로 이어지는 과정을 봅니다.',
  promotionTrial: '고정된 Candidate와 Champion을 같은 조건에서 평가합니다.',
  connection: '시장 데이터 제공자 인증과 연결 상태를 관리합니다.',
  system: '프로세스, 자원, 오류와 로그를 진단합니다.',
  experts: '실제 Registry의 전문가 적재와 추론 상태를 비교합니다.',
  'trading-moe': '독립 TradingMoE worker의 판단부터 optimizer까지 확인합니다.',
  assembly: '조립 recipe를 생성하고 후보 시험을 순환합니다.',
};

export default function App() {
  const page = pageFromHash(useHash()),
    status = usePolling('/api/status'),
    s = status.data,
    title = pages.find(([key]) => key === page)?.[1] || '운영';
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [page]);

  let content;
  if (page === 'assembly') content = <Assembly />;
  else if (page === 'promotionTrial') content = <PromotionTrial />;
  else if (page === 'experts') content = <Experts />;
  else if (page === 'trading-moe') content = <TradingMoE />;
  else if (page === 'connection') content = <Connection />;
  else if (!s) content = status.error ? <ErrorState message={status.error} retry={status.refresh} /> : <Loading />;
  else if (page === 'markets') content = <Markets status={s} />;
  else if (page === 'learning') content = <Learning status={s} />;
  else if (page === 'system') content = <Details status={s} />;
  else content = <Operations status={s} />;

  return (
    <CommandProvider>
      <div className="workspace">
        <aside className="rail">
          <a className="brand" href="#control"><span className="brand-mark">M</span><span>MARKET<br />WORKS</span></a>
          <div className="rail-caption">TRADING SYSTEM</div>
          <nav className="primary-nav" aria-label="주요 메뉴">
            {pages.map(([key, name], index) => (
              <a key={key} href={'#' + key} aria-current={key === page ? 'page' : undefined}>
                <span className="nav-index">{String(index + 1).padStart(2, '0')}</span><span>{name}</span>
              </a>
            ))}
          </nav>
          <div className="rail-footer"><span className="live-dot" />Paper operations</div>
        </aside>
        <main className="main-area">
          <header className="global-bar">
            <div className="global-state" aria-label="전역 시스템 상태">
              <Signal label="웹서버" value={status.error ? '연결 오류' : s ? '연결됨' : '연결 중'} tone={status.error ? 'bad' : s ? 'good' : 'neutral'} />
              <Signal label="시장 Feed" value={s?.feed_running ? '수신 중' : '정지'} tone={s?.feed_running ? 'good' : 'neutral'} />
              <Signal label="실제 주문" value={s?.real_orders_enabled ? '허용' : '차단'} tone={s?.real_orders_enabled ? 'bad' : 'good'} />
            </div>
            <span className="last-refresh">갱신 {status.updatedAt ? time(new Date(status.updatedAt).toISOString()) : '—'}</span>
          </header>
          <div className="mobile-brand"><a href="#control">MARKET WORKS</a><span>TRADING SYSTEM · PAPER</span></div>
          <nav className="mobile-nav" aria-label="주요 메뉴">
            {pages.map(([key, name], index) => <a key={key} href={'#' + key} aria-current={key === page ? 'page' : undefined}><small>{String(index + 1).padStart(2, '0')}</small>{name}</a>)}
          </nav>
          <section className="page-heading">
            <div><div className="page-kicker">PAPER OPERATIONS / {String(page).toUpperCase()}</div><h1>{title}</h1><p>{descriptions[page]}</p></div>
            <div className="page-heading-badge"><span className="live-dot" />{s?.running ? '공유 runtime 실행 중' : 'PAPER · 실제 주문 차단'}</div>
          </section>
          {status.error && s && <ErrorState message={'상태 갱신 오류 · 마지막 수신값 표시 중: ' + status.error} retry={status.refresh} />}
          <div className="page-view" key={page}>{content}</div>
        </main>
      </div>
    </CommandProvider>
  );
}
