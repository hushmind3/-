import type { ApiResult, Data, ReadEndpoint, Responses } from './types';
export const endpoints = {
  start: '/api/start',
  stop: '/api/stop',
  reconnect: '/api/feed/reconnect',
  serverRestart: '/api/server/restart',
  resetAccounts: '/api/paper-accounts/reset',
  modes: '/api/modes',
  connect: '/api/provider/connect',
  testProvider: '/api/provider/test',
  publicIp: '/api/provider/public-ip',
  expertFusion: '/api/experts/fusion',
} as const;
export async function request<T>(path: string, body?: Data, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    cache: 'no-store',
    signal,
  });
  const data: ApiResult = await response.json();
  if (!response.ok || data.ok === false)
    throw new Error(data.error || data.message || `HTTP ${response.status}`);
  return data as T;
}
export const api = {
  get: <P extends ReadEndpoint>(path: P, signal?: AbortSignal) =>
    request<Responses[P]>(path, undefined, signal),
  command: (path: string, body: Data = {}) => request<ApiResult>(path, body),
  raw: (id: string) => request<Data>('/api/experts/output?id=' + encodeURIComponent(id)),
  fusion: () => request<Data>(endpoints.expertFusion),
  publicIp: () => request<Data>(endpoints.publicIp),
  model: (role: 'champion' | 'candidate', action: 'start' | 'stop') =>
    '/api/models/' + role + '/' + action,
  moe: (action: 'start' | 'stop') => '/api/trading-moe/' + action,
  assembly: (action: string) => '/api/assembly/' + action,
};
