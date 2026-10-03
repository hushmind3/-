import { createContext, useContext, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { api } from '../api';
import { refreshAll } from '../hooks';
import { obj, str } from '../data';
import type { Data, Json } from '../types';

type Tone = 'good' | 'warn' | 'bad' | 'neutral';

export function Signal({ label, value, tone = 'neutral' }: { label: string; value: ReactNode; tone?: Tone }) {
  return <span className={'signal ' + tone}><i aria-hidden="true" /><span className="signal-label">{label}</span><strong>{value}</strong></span>;
}

export function Action({ tone = 'neutral', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone }) {
  return <button {...props} className={'action ' + tone + ' ' + (props.className || '')} />;
}

export function DataTable({ headers, rows, empty = '표시할 항목이 없습니다.' }: { headers: string[]; rows: ReactNode[][]; empty?: string }) {
  return <div className="data-table-scroll"><table className="data-table"><thead><tr>{headers.map((header) => <th scope="col" key={header}>{header}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>) : <tr><td colSpan={headers.length} className="table-empty">{empty}</td></tr>}</tbody></table></div>;
}

export function Disclosure({ title, children, className = '' }: { title: ReactNode; children: ReactNode; className?: string }) {
  const [open,setOpen]=useState(false);
  return <details className={'disclosure ' + className} onToggle={(event)=>setOpen(event.currentTarget.open)}><summary>{title}</summary>{open&&<div className="disclosure-content">{children}</div>}</details>;
}

export function SectionLabel({ eyebrow, title, detail }: { eyebrow?: string; title: string; detail?: ReactNode }) {
  return <header className="section-label"><div>{eyebrow && <small>{eyebrow}</small>}<h2>{title}</h2></div>{detail && <div className="section-note">{detail}</div>}</header>;
}

export function StageTrack({ steps }: { steps: { label: string; detail?: ReactNode; state: 'done' | 'active' | 'waiting' | 'blocked' }[] }) {
  return <ol className="stage-track">{steps.map((step, index) => <li key={step.label} className={'stage ' + step.state}><span className="stage-number">{String(index + 1).padStart(2, '0')}</span><div><strong>{step.label}</strong>{step.detail && <small>{step.detail}</small>}</div></li>)}</ol>;
}

export function JsonDetails({ title, data }: { title: string; data: Json | undefined }) {
  if (data == null) return null;
  return <Disclosure title={title}><pre className="json-view">{JSON.stringify(data, null, 2)}</pre></Disclosure>;
}

export function Loading() { return <div className="empty-state" role="status">상태 불러오는 중…</div>; }
export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="inline-error" role="alert"><span>{message}</span>{retry && <Action onClick={retry}>다시 조회</Action>}</div>;
}

interface CommandState { pending: string | null; run: (key: string, label: string, task: () => Promise<unknown>) => Promise<boolean> }
const CommandContext = createContext<CommandState | null>(null);
export function CommandProvider({ children }: { children: ReactNode }) {
  const busy = useRef(false), [pending, setPending] = useState<string | null>(null), [notice, setNotice] = useState<{ message: string; failed: boolean }>();
  async function run(key: string, label: string, task: () => Promise<unknown>) {
    if (busy.current) return false;
    busy.current = true; setPending(key); setNotice({ message: label + ' 요청 중…', failed: false });
    try {
      const result = obj((await task()) as Json);
      setNotice({ message: str(result.message, '변경 적용 완료'), failed: false });
      await refreshAll();
      return true;
    } catch (error) {
      setNotice({ message: error instanceof Error ? error.message : String(error), failed: true });
      return false;
    } finally { busy.current = false; setPending(null); }
  }
  return <CommandContext.Provider value={{ pending, run }}>{children}{notice && <div className={'command-toast ' + (notice.failed ? 'bad' : 'good')} role={notice.failed ? 'alert' : 'status'}><span>{notice.message}</span><Action aria-label="알림 닫기" onClick={() => setNotice(undefined)}>×</Action></div>}</CommandContext.Provider>;
}
function useCommands() { const context = useContext(CommandContext); if (!context) throw new Error('CommandProvider missing'); return context; }

export function ActionButton({ label, path, body = {}, task, disabled, confirm, tone = 'neutral', pendingLabel }: { label: string; path?: string; body?: Data; task?: () => Promise<unknown>; disabled?: boolean; confirm?: string; tone?: Tone; pendingLabel?: string }) {
  const { pending, run } = useCommands(), key = path || label;
  return <Action tone={tone} disabled={disabled || pending !== null} aria-busy={pending === key} onClick={() => { if (confirm && !window.confirm(confirm)) return; void run(key, label, task || (() => api.command(path!, body))); }}>{pending === key ? pendingLabel || label + ' 요청 중…' : label}</Action>;
}

export function Toggle({ label, enabled, path, field }: { label: string; enabled: boolean; path: string; field: string }) {
  const { pending, run } = useCommands(), key = path + '/' + field;
  return <Action role="switch" aria-checked={enabled} className={enabled ? 'selected' : ''} disabled={pending !== null} aria-busy={pending === key} onClick={() => void run(key, label, () => api.command(path, { [field]: !enabled }))}>{label}: <b>{pending === key ? '변경 중' : enabled ? 'ON' : 'OFF'}</b></Action>;
}
