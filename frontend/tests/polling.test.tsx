import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { usePolling } from '../src/hooks';
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
function Consumer() {
  const q = usePolling('/api/health', 1000);
  return <div>{q.loading ? 'loading' : q.error || String(q.data?.value)}</div>;
}
it('shares one poll between consumers, schedules completion, stops after unmount', async () => {
  vi.useFakeTimers();
  const f = vi.fn(async () => new Response('{"value":1}'));
  vi.stubGlobal('fetch', f);
  const r = render(
    <>
      <Consumer />
      <Consumer />
    </>,
  );
  await act(async () => {});
  expect(f).toHaveBeenCalledTimes(1);
  r.rerender(
    <>
      <Consumer />
      <Consumer />
    </>,
  );
  await act(async () => {});
  expect(f).toHaveBeenCalledTimes(1);
  await act(async () => vi.advanceTimersByTimeAsync(1000));
  expect(f).toHaveBeenCalledTimes(2);
  r.unmount();
  await act(async () => vi.advanceTimersByTimeAsync(3000));
  expect(f).toHaveBeenCalledTimes(2);
});
it('retains last response on error and recovers next cycle', async () => {
  vi.useFakeTimers();
  let failed = false;
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      if (failed) throw Error('offline');
      return new Response('{"value":2}');
    }),
  );
  render(<Consumer />);
  await act(async () => {});
  expect(screen.getByText('2')).toBeTruthy();
  failed = true;
  await act(async () => vi.advanceTimersByTimeAsync(1000));
  expect(screen.getByText('offline')).toBeTruthy();
  failed = false;
  await act(async () => vi.advanceTimersByTimeAsync(1000));
  expect(screen.getByText('2')).toBeTruthy();
});
it('pauses hidden-tab polls and resumes on visibility change', async () => {
  vi.useFakeTimers();
  const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
  const f = vi.fn(async () => new Response('{"value":3}'));
  vi.stubGlobal('fetch', f);
  render(<Consumer />);
  await act(async () => {});
  expect(f).toHaveBeenCalledTimes(1);
  hidden.mockReturnValue(true);
  await act(async () => vi.advanceTimersByTimeAsync(5000));
  expect(f).toHaveBeenCalledTimes(1);
  hidden.mockReturnValue(false);
  await act(async () => document.dispatchEvent(new Event('visibilitychange')));
  expect(f).toHaveBeenCalledTimes(2);
});
it('does not overlap slow requests', async () => {
  vi.useFakeTimers();
  let resolve: (response: Response) => void = () => {};
  const f = vi.fn(() => new Promise<Response>((r) => (resolve = r)));
  vi.stubGlobal('fetch', f);
  render(<Consumer />);
  await act(async () => vi.advanceTimersByTimeAsync(4000));
  expect(f).toHaveBeenCalledTimes(1);
  await act(async () => resolve(new Response('{"value":4}')));
  await act(async () => vi.advanceTimersByTimeAsync(1000));
  expect(f).toHaveBeenCalledTimes(2);
});
