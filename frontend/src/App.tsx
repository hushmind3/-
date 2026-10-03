import { useEffect } from 'react';
import { CommandProvider, ErrorState, Loading, Badge } from './components/ui';
import { useHash, usePolling } from './hooks';
import { pageFromHash, pages, time } from './data';
import { Operations, Markets } from './pages/Operations';
import { Learning } from './pages/Learning';
import { Trial } from './pages/Trial';
import { Connection } from './pages/Connection';
import { Details } from './pages/Details';
import { Experts } from './pages/Experts';
import { TradingMoE } from './pages/TradingMoE';
import { Assembly } from './pages/Assembly';
export default function App() {
  const page = pageFromHash(useHash()),
    q = usePolling('/api/status'),
    s = q.data,
    title = pages.find(([p]) => p === page)?.[1];
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [page]);
  let content;
  if (page === 'assembly') content = <Assembly />;
  else if (page === 'experts') content = <Experts />;
  else if (page === 'trading-moe') content = <TradingMoE />;
  else if (page === 'connection') content = <Connection />;
  else if (!s) content = q.error ? <ErrorState message={q.error} retry={q.refresh} /> : <Loading />;
  else
    content =
      page === 'markets' ? (
        <Markets status={s} />
      ) : page === 'learning' ? (
        <Learning status={s} />
      ) : page === 'promotionTrial' ? (
        <Trial status={s} />
      ) : page === 'system' ? (
        <Details status={s} />
      ) : (
        <Operations status={s} />
      );
  return (
    <CommandProvider>
      <div className="app-shell">
        <aside className="sidebar">
          <a className="brand" href="#control">
            금융매매모델
          </a>
          <span className="muted">TradingMoE · PAPER</span>
          <nav aria-label="화면 선택">
            {pages.map(([key, name]) => (
              <a key={key} href={'#' + key} aria-current={key === page ? 'page' : undefined}>
                {name}
              </a>
            ))}
          </nav>
        </aside>
        <main>
          <header className="page-header">
            <div>
              <h1>{title}</h1>
              <p className="muted">
                상태 조회 {q.updatedAt ? time(new Date(q.updatedAt).toISOString()) : '연결 중'}
              </p>
            </div>
            <div className="header-badges">
              <Badge tone={q.error ? 'bad' : s ? 'good' : 'neutral'}>
                {q.error ? '서버 연결 오류' : s ? '웹서버 연결' : '연결 중'}
              </Badge>
              <Badge tone={s?.feed_running ? 'good' : 'neutral'}>
                Feed {s?.feed_running ? 'ON' : 'OFF'}
              </Badge>
              <Badge tone={s?.real_orders_enabled ? 'bad' : 'neutral'}>
                실제 주문 {s?.real_orders_enabled ? 'ON' : 'OFF'}
              </Badge>
            </div>
          </header>
          {q.error && s && (
            <ErrorState message={'연결 오류 · 이전 상태 표시 중: ' + q.error} retry={q.refresh} />
          )}
          <div className="page-content" key={page}>
            {content}
          </div>
        </main>
      </div>
    </CommandProvider>
  );
}
