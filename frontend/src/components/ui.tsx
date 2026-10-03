import {
  createContext,
  memo,
  useContext,
  useRef,
  useState,
  type ReactNode,
  type ButtonHTMLAttributes,
} from 'react';
import { api } from '../api';
import { refreshAll } from '../hooks';
import { bytes, fmt, label, obj, pct, state, str, time } from '../data';
import type { Data, Json } from '../types';
type Tone = 'good' | 'warn' | 'bad' | 'neutral';
export const Badge = memo(
  ({ children, tone = 'neutral' }: { children: ReactNode; tone?: Tone }) => (
    <span className={'badge ' + tone}>{children}</span>
  ),
);
export const Card = memo(
  ({ title, children, badge }: { title?: ReactNode; children: ReactNode; badge?: ReactNode }) => (
    <article className="card">
      {title && (
        <header className="card-head">
          <h2>{title}</h2>
          {badge}
        </header>
      )}
      {children}
    </article>
  ),
);
export const StatCard = memo(
  ({
    title,
    value,
    detail,
    tone,
    badge,
    progress,
  }: {
    title: string;
    value: ReactNode;
    detail?: ReactNode;
    tone?: Tone;
    badge?: ReactNode;
    progress?: number;
  }) => (
    <Card title={title} badge={badge}>
      <strong className={'stat ' + (tone || '')}>{value}</strong>
      {progress != null && (
        <progress aria-label={title} value={Math.max(0, Math.min(100, progress))} max={100} />
      )}
      {detail && <p className="muted">{detail}</p>}
    </Card>
  ),
);
export function Stats({ children }: { children: ReactNode }) {
  return <div className="stats-grid">{children}</div>;
}
export function Buttons({ children }: { children: ReactNode }) {
  return <div className="buttons">{children}</div>;
}
export function Button({
  children,
  tone = 'neutral',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone }) {
  return (
    <button {...props} className={'button ' + tone + ' ' + (props.className || '')}>
      {children}
    </button>
  );
}
export function Empty({ children = '기록 없음' }: { children?: ReactNode }) {
  return <p className="empty">{children}</p>;
}
export function Loading() {
  return (
    <p className="empty" role="status">
      상태 불러오는 중…
    </p>
  );
}
export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div className="notice bad" role="alert">
      {message}
      {retry && <Button onClick={retry}>다시 조회</Button>}
    </div>
  );
}
export function Panel({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details className="panel" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{title}</summary>
      {open && <div className="panel-body">{children}</div>}
    </details>
  );
}
export const Table = memo(
  ({
    headers,
    rows: values,
    empty = '기록 없음',
  }: {
    headers: string[];
    rows: ReactNode[][];
    empty?: string;
  }) => (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {values.length ? (
            values.map((r, i) => (
              <tr key={i}>
                {r.map((v, j) => (
                  <td key={j}>{v}</td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={headers.length}>
                <Empty>{empty}</Empty>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  ),
);
export function Display({
  value,
  field = '',
}: {
  value: Json | undefined;
  field?: string;
}): ReactNode {
  if (value == null) return '—';
  if (typeof value === 'boolean')
    return <Badge tone={value ? 'good' : 'neutral'}>{value ? '예' : '아니오'}</Badge>;
  if (Array.isArray(value)) return value.length ? <DataTree data={value} /> : <Empty />;
  if (typeof value === 'object') return <DataTree data={value} />;
  if (field === 'status' || field === 'evaluation_state') return state(value);
  if (/(_bytes|^bytes$)/.test(field)) return bytes(value);
  if (/return|weight$|coverage|drawdown/.test(field) && typeof value === 'number')
    return pct(value);
  if (/timestamp|_at$|_utc$/.test(field) && typeof value === 'string') return time(value);
  return typeof value === 'number' ? fmt(value, 6) : String(value);
}
export function DataTree({ data }: { data: Json | undefined }) {
  const [limit, setLimit] = useState(30);
  if (data == null) return <Empty />;
  if (Array.isArray(data))
    return (
      <>
        {data.slice(0, limit).map((v, i) => (
          <div key={i} className="data-row">
            <DataTree data={v} />
          </div>
        ))}
        {data.length > limit && (
          <Button onClick={() => setLimit((n) => n + 50)}>더 보기 ({data.length - limit}개)</Button>
        )}
      </>
    );
  if (typeof data !== 'object') return <span>{str(data)}</span>;
  const entries = Object.entries(data);
  if (!entries.length) return <Empty />;
  return (
    <dl className="data-list">
      {entries.map(([key, value]) => (
        <div key={key}>
          {typeof value === 'object' && value !== null ? (
            <Panel title={label(key) + (Array.isArray(value) ? ` · ${value.length}개` : '')}>
              <DataTree data={value} />
            </Panel>
          ) : (
            <>
              <dt>{label(key)}</dt>
              <dd>
                <Display value={value} field={key} />
              </dd>
            </>
          )}
        </div>
      ))}
    </dl>
  );
}
export function DataPanel({ title, data }: { title: string; data: Json | undefined }) {
  return (
    <Panel title={title}>
      <DataTree data={data} />
    </Panel>
  );
}
interface CommandContext {
  pending: string | null;
  run: (key: string, label: string, task: () => Promise<unknown>) => Promise<boolean>;
}
const Commands = createContext<CommandContext | null>(null);
export function CommandProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<string | null>(null);
  const busy = useRef(false);
  const [notice, setNotice] = useState<{ text: string; error: boolean }>();
  async function run(key: string, description: string, task: () => Promise<unknown>) {
    if (busy.current) return false;
    busy.current = true;
    setPending(key);
    setNotice({ text: description + ' 요청 중…', error: false });
    try {
      const result = obj((await task()) as Json);
      setNotice({ text: str(result.message, '요청 적용 완료'), error: false });
      await refreshAll();
      return true;
    } catch (e) {
      setNotice({
        text: e instanceof Error ? e.message : String(e),
        error: true,
      });
      return false;
    } finally {
      busy.current = false;
      setPending(null);
    }
  }
  return (
    <Commands.Provider value={{ pending, run }}>
      {children}
      {notice && (
        <div
          className={'toast ' + (notice.error ? 'bad' : 'good')}
          role={notice.error ? 'alert' : 'status'}
        >
          {notice.text}
          <Button aria-label="알림 닫기" onClick={() => setNotice(undefined)}>
            ×
          </Button>
        </div>
      )}
    </Commands.Provider>
  );
}
function useCommand() {
  const c = useContext(Commands);
  if (!c) throw new Error('CommandProvider missing');
  return c;
}
export function ActionButton({
  label: description,
  path,
  body = {},
  task,
  disabled,
  confirm,
  tone = 'neutral',
  pendingLabel,
}: {
  label: string;
  path?: string;
  body?: Data;
  task?: () => Promise<unknown>;
  disabled?: boolean;
  confirm?: string;
  tone?: Tone;
  pendingLabel?: string;
}) {
  const { pending, run } = useCommand();
  const key = path || description;
  return (
    <Button
      tone={tone}
      disabled={disabled || pending !== null}
      aria-busy={pending === key}
      onClick={() => {
        if (confirm && !window.confirm(confirm)) return;
        void run(key, description, task || (() => api.command(path!, body)));
      }}
    >
      {pending === key ? pendingLabel || description + ' 요청 중…' : description}
    </Button>
  );
}
export function Toggle({
  label: description,
  enabled,
  path,
  field,
}: {
  label: string;
  enabled: boolean;
  path: string;
  field: string;
}) {
  const { pending, run } = useCommand();
  const key = path + '/' + field;
  return (
    <Button
      className={enabled ? 'active' : ''}
      role="switch"
      aria-checked={enabled}
      disabled={pending !== null}
      aria-busy={pending === key}
      onClick={() => void run(key, description, () => api.command(path, { [field]: !enabled }))}
    >
      {description}: {pending === key ? '변경 중…' : enabled ? 'ON · 끄기' : 'OFF · 켜기'}
    </Button>
  );
}
