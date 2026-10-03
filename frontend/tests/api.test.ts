import { afterEach, expect, it, vi } from 'vitest';
import { api, request } from '../src/api';
import { pageFromHash, bytes, pct, num } from '../src/data';
afterEach(() => vi.unstubAllGlobals());
it('central API sends typed POST payload', async () => {
  const f = vi.fn(async () => new Response('{"ok":true}'));
  vi.stubGlobal('fetch', f);
  await api.command('/api/modes', { learning_enabled: false });
  expect(f.mock.calls[0]).toEqual([
    '/api/modes',
    expect.objectContaining({
      method: 'POST',
      body: '{"learning_enabled":false}',
      cache: 'no-store',
    }),
  ]);
});
it('reports backend failure even with HTTP 200', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response('{"ok":false,"error":"test failure"}')),
  );
  await expect(request('/api/test', {})).rejects.toThrow('test failure');
});
it('reports HTTP errors', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response('{"error":"not found"}', { status: 404 })),
  );
  await expect(request('/api/test')).rejects.toThrow('not found');
});
it('encodes expert IDs', async () => {
  const f = vi.fn(async () => new Response('{}'));
  vi.stubGlobal('fetch', f);
  await api.raw('slope/1');
  expect(f.mock.calls[0]).toEqual(['/api/experts/output?id=slope%2F1', expect.anything()]);
});
it('keeps all hash aliases and finite formatters', () => {
  expect(pageFromHash('#assembly')).toBe('assembly');
  expect(pageFromHash('#overview')).toBe('control');
  expect(pageFromHash('#details')).toBe('system');
  expect(pageFromHash('#unknown')).toBe('control');
  expect(bytes(1024 ** 3)).toBe('1 GiB');
  expect(pct(0.2)).toBe('20%');
  expect(num('bad')).toBe(0);
});
