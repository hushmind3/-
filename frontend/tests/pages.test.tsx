import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from '../src/App';
import { pages } from '../src/data';
import { responses } from './fixtures';
import type { Data } from '../src/types';
let fixtures: Record<string, Data>;
let posts: { path: string; body: Data }[];
beforeEach(() => {
  vi.stubGlobal('scrollTo', vi.fn());
  fixtures = structuredClone(responses);
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        const body: Data = JSON.parse(String(init.body));
        posts.push({ path, body });
        if (path === '/api/modes') Object.assign(fixtures['/api/status'], body);
        if (path.startsWith('/api/models/')) {
          const [, role, action] = path.match(/models\/(champion|candidate)\/(start|stop)/)!;
          const runtime = fixtures['/api/status'].model_runtime as Data;
          (runtime[role] as Data).loaded = action === 'start';
          (runtime[role] as Data).requested = action === 'start';
          (runtime[role] as Data).status = action === 'start' ? 'running' : 'stopped';
        }
        if (path.startsWith('/api/trading-moe/')) {
          fixtures['/api/trading-moe/status'].alive = path.endsWith('start');
          fixtures['/api/trading-moe/status'].status = path.endsWith('start')
            ? 'running'
            : 'stopped';
        }
        if (path === '/api/assembly/settings')
          Object.assign(fixtures['/api/assembly/status'].settings as Data, body);
        if (path === '/api/assembly/start' || path === '/api/assembly/stop')
          fixtures['/api/assembly/status'].enabled = path.endsWith('start');
        if (path.includes('/assembly/trial/'))
          (fixtures['/api/assembly/status'].worker as Data).alive = path.endsWith('start');
        return new Response(JSON.stringify({ ok: true, message: '적용 완료' }));
      }
      return new Response(JSON.stringify(fixtures[path] || { ok: true }));
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
async function open(page: string) {
  window.location.hash = page;
  render(<App />);
  await screen.findByRole('heading', { name: pages.find(([p]) => p === page)?.[1], level: 1 });
  await waitFor(() => expect(screen.queryByText('상태 불러오는 중…')).toBeNull());
}
async function click(name: string) {
  fireEvent.click(await screen.findByRole('button', { name }));
  await waitFor(() =>
    expect(screen.getByRole('button', { name }).getAttribute('aria-busy')).not.toBe('true'),
  );
}
describe('all existing pages', () => {
  it.each(pages)('%s renders without console errors', async (page) => {
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    await open(page);
    expect(
      screen.getByRole('heading', { name: pages.find(([p]) => p === page)?.[1], level: 1 }),
    ).toBeTruthy();
    expect(err).not.toHaveBeenCalled();
  });
  it('preserves live hash navigation', async () => {
    await open('control');
    fireEvent.click(screen.getByRole('link', { name: '조립 · 자동실험' }));
    window.location.hash = 'assembly';
    fireEvent(window, new HashChangeEvent('hashchange'));
    await screen.findByRole('heading', { name: '조립 · 자동실험', level: 1 });
  });
  it('filters search, fresh quotes and market without losing controls', async () => {
    await open('markets');
    fireEvent.change(screen.getByRole('textbox', { name: '종목 검색' }), {
      target: { value: 'MSFT' },
    });
    expect(screen.getAllByText('MSFT').length).toBeGreaterThan(0);
    expect(screen.queryByText('005930.KS')).toBeNull();
    fireEvent.change(screen.getByRole('textbox', { name: '종목 검색' }), { target: { value: '' } });
    fireEvent.click(screen.getByRole('checkbox', { name: '최근 5분 수신만' }));
    expect(screen.queryByText('005930.KS')).toBeNull();
  });
  it('preserves per-timeframe samples and does not call missing records zero input', async () => {
    await open('learning');
    expect(screen.getByText('90%')).toBeTruthy();
    expect(screen.getByText('128')).toBeTruthy();
    expect(screen.getByText('64')).toBeTruthy();
  });
  it('treats string logs as lines', async () => {
    await open('system');
    expect(screen.getByText('server ready')).toBeTruthy();
  });
});
describe('existing control endpoints and independent flags', () => {
  it.each(['observe_enabled', 'paper_enabled', 'learning_enabled'])(
    'posts only %s',
    async (field) => {
      await open('control');
      const i = ['observe_enabled', 'paper_enabled', 'learning_enabled'].indexOf(field),
        label = ['모델 판단', '가상 체결', '경험 학습'][i];
      fireEvent.click(screen.getByRole('switch', { name: new RegExp(label + ':') }));
      await waitFor(() =>
        expect(posts).toContainEqual({ path: '/api/modes', body: { [field]: false } }),
      );
      await waitFor(() =>
        expect(
          screen
            .getByRole('switch', { name: new RegExp(label + ':') })
            .getAttribute('aria-checked'),
        ).toBe('false'),
      );
      expect(Object.keys(posts[0].body)).toEqual([field]);
    },
  );
  it.each(['champion', 'candidate'])('starts and stops %s alone', async (role) => {
    await open('control');
    const name = role === 'champion' ? 'Champion' : 'Candidate';
    await click(name + ' 시작');
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: name + ' 저장 후 정지' }).hasAttribute('disabled'),
      ).toBe(false),
    );
    await click(name + ' 저장 후 정지');
    expect(posts.filter((p) => p.path.includes('/models/')).map((p) => p.path)).toEqual([
      '/api/models/' + role + '/start',
      '/api/models/' + role + '/stop',
    ]);
  });
  it('system commands preserve feed-only startup and reset scope', async () => {
    await open('control');
    await click('시장 Feed 시작');
    await click('시세 재연결');
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    await click('두 장기 운영계좌 초기화');
    expect(posts.some((p) => p.path.includes('reset'))).toBe(false);
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    await click('두 장기 운영계좌 초기화');
    await click('웹서버만 재시작');
    await click('전체 정지');
    expect(posts.map((p) => p.path)).toEqual([
      '/api/start',
      '/api/feed/reconnect',
      '/api/paper-accounts/reset',
      '/api/server/restart',
      '/api/stop',
    ]);
    expect(posts[0].body).toEqual({ mode: 'live' });
  });
  it('starts and stops dedicated MoE worker', async () => {
    await open('trading-moe');
    await click('TradingMoE 시작');
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: 'TradingMoE 저장 후 정지' }).hasAttribute('disabled'),
      ).toBe(false),
    );
    await click('TradingMoE 저장 후 정지');
    expect(posts.map((p) => p.path)).toEqual(['/api/trading-moe/start', '/api/trading-moe/stop']);
    expect(screen.getByText('0.759')).toBeTruthy();
    expect(screen.getByText('1,583.1983')).toBeTruthy();
  });
  it('all assembly actions and settings use existing endpoints', async () => {
    await open('assembly');
    await click('자동조립 시작');
    await click('자동조립 정지');
    await click('새 Candidate 즉시 생성');
    await click('현재 Candidate 시험 시작');
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: '현재 Candidate 시험 정지' }).hasAttribute('disabled'),
      ).toBe(false),
    );
    await click('현재 Candidate 시험 정지');
    await click('현재 Candidate 탈락 · 다음 후보');
    for (const name of ['자동 Candidate 교체', '자동 승격', '새 Expert 자동감지']) {
      fireEvent.click(screen.getByRole('switch', { name: new RegExp(name + ':') }));
      await waitFor(() =>
        expect(
          screen.getByRole('switch', { name: new RegExp(name + ':') }).getAttribute('aria-busy'),
        ).not.toBe('true'),
      );
    }
    expect(posts.slice(0, 6).map((p) => p.path)).toEqual(
      ['start', 'stop', 'generate', 'trial/start', 'trial/stop', 'next'].map(
        (p) => '/api/assembly/' + p,
      ),
    );
    expect(posts.slice(6).map((p) => p.body)).toEqual([
      { auto_replace: false },
      { auto_promote: true },
      { detect_experts: false },
    ]);
    expect(screen.getByText('Registry Dynamic Expert')).toBeTruthy();
    expect(screen.getByText('NEW')).toBeTruthy();
  });
  it('connects provider, clears keys, preserves stored test and public IP', async () => {
    await open('connection');
    fireEvent.change(screen.getByLabelText('App Key'), { target: { value: 'test-key' } });
    fireEvent.change(screen.getByLabelText('Secret'), { target: { value: 'test-secret' } });
    await click('인증 정보 저장 · 연결');
    expect((screen.getByLabelText('App Key') as HTMLInputElement).value).toBe('');
    await click('저장된 인증으로 재확인');
    await click('공인 IP 조회');
    expect(posts[0]).toEqual({
      path: '/api/provider/connect',
      body: { environment: 'real', account: '', app_key: 'test-key', secret: 'test-secret' },
    });
    expect(posts[1].path).toBe('/api/provider/test');
    expect(screen.getByText('127.0.0.1')).toBeTruthy();
  });
  it('loads raw expert output only on request and keeps fusion download', async () => {
    await open('experts');
    expect(screen.queryByText('0.1')).toBeNull();
    fireEvent.click(screen.getByText('원본 입력 · 출력 shape · raw 출력'));
    await screen.findByRole('button', { name: 'Registry Dynamic Expert 원본 출력 조회' });
    await click('Registry Dynamic Expert 원본 출력 조회');
    expect(
      vi.mocked(fetch).mock.calls.some(([p]) => String(p).includes('output?id=test_expert')),
    ).toBe(true);
    await click('최근 통합 출력 조회');
    expect(screen.getByRole('link', { name: '전체 통합 JSON 저장' }).getAttribute('href')).toBe(
      '/api/experts/fusion',
    );
  });
});
