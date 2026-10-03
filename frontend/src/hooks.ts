import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { api } from './api';
import type { Data, ReadEndpoint, Responses } from './types';
interface Snapshot {
  data?: Data;
  error?: string;
  loading: boolean;
  updatedAt?: number;
}
interface Resource {
  snapshot: Snapshot;
  listeners: Set<() => void>;
  timer?: ReturnType<typeof setTimeout>;
  controller?: AbortController;
  promise?: Promise<void>;
  interval: number;
}
const resources = new Map<ReadEndpoint, Resource>();
function resource(path: ReadEndpoint) {
  let r = resources.get(path);
  if (!r) {
    r = { snapshot: { loading: true }, listeners: new Set(), interval: 5000 };
    resources.set(path, r);
  }
  return r;
}
function publish(r: Resource, snapshot: Snapshot) {
  r.snapshot = snapshot;
  r.listeners.forEach((l) => l());
}
async function refresh(path: ReadEndpoint): Promise<void> {
  const r = resource(path);
  if (r.promise) return r.promise;
  clearTimeout(r.timer);
  r.controller = new AbortController();
  r.promise = (async () => {
    try {
      const data = await api.get(path, r.controller?.signal);
      publish(r, { data, loading: false, updatedAt: Date.now() });
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError'))
        publish(r, {
          ...r.snapshot,
          loading: false,
          error: e instanceof Error ? e.message : String(e),
        });
    } finally {
      r.promise = undefined;
      if (r.listeners.size)
        r.timer = setTimeout(() => {
          if (!document.hidden) void refresh(path);
          else schedule(path);
        }, r.interval);
    }
  })();
  return r.promise;
}
function schedule(path: ReadEndpoint) {
  const r = resource(path);
  clearTimeout(r.timer);
  r.timer = setTimeout(() => {
    if (!document.hidden) void refresh(path);
    else schedule(path);
  }, r.interval);
}
export function usePolling<P extends ReadEndpoint>(path: P, interval = 5000) {
  const r = resource(path);
  r.interval = interval;
  const subscribe = useCallback(
    (listener: () => void) => {
      r.listeners.add(listener);
      if (r.listeners.size === 1) {
        if (r.promise && r.controller?.signal.aborted) void r.promise.then(() => refresh(path));
        else void refresh(path);
      }
      return () => {
        r.listeners.delete(listener);
        if (!r.listeners.size) {
          clearTimeout(r.timer);
          r.controller?.abort();
        }
      };
    },
    [path, r],
  );
  const snapshot = useSyncExternalStore(subscribe, () => r.snapshot);
  useEffect(() => {
    const visible = () => {
      if (!document.hidden) {
        clearTimeout(r.timer);
        void refresh(path);
      }
    };
    document.addEventListener('visibilitychange', visible);
    return () => document.removeEventListener('visibilitychange', visible);
  }, [path, r]);
  return {
    ...snapshot,
    data: snapshot.data as Responses[P] | undefined,
    refresh: () => refresh(path),
  };
}
export async function refreshAll() {
  await Promise.all(
    [...resources]
      .filter(([, r]) => r.listeners.size)
      .map(async ([p, r]) => {
        clearTimeout(r.timer);
        if (r.promise) await r.promise;
        return refresh(p);
      }),
  );
}
function subscribeHash(onChange: () => void) {
  window.addEventListener('hashchange', onChange);
  window.addEventListener('popstate', onChange);
  return () => {
    window.removeEventListener('hashchange', onChange);
    window.removeEventListener('popstate', onChange);
  };
}
function currentHash() {
  return window.location.hash;
}
export function useHash() {
  return useSyncExternalStore(subscribeHash, currentHash, () => '#control');
}
