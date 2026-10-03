import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { ActionButton, Button, CommandProvider, Panel, Toggle } from '../src/components/ui';
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
it('updates content without recreating the button DOM node', () => {
  const r = render(<Button>1</Button>);
  const original = screen.getByRole('button');
  r.rerender(<Button>2</Button>);
  expect(screen.getByRole('button')).toBe(original);
  expect(original.textContent).toBe('2');
});
it('does not mount hidden details until expanded', async () => {
  render(
    <Panel title="상세">
      <div>expensive data</div>
    </Panel>,
  );
  expect(screen.queryByText('expensive data')).toBeNull();
  fireEvent.click(screen.getByText('상세'));
  await screen.findByText('expensive data');
});
it('prevents duplicate commands and shows pending then success', async () => {
  let resolve: (v: unknown) => void = () => {};
  const task = vi.fn(() => new Promise((r) => (resolve = r)));
  render(
    <CommandProvider>
      <ActionButton label="시작" task={task} />
    </CommandProvider>,
  );
  fireEvent.click(screen.getByRole('button', { name: '시작' }));
  fireEvent.click(screen.getByRole('button', { name: '시작 요청 중…' }));
  expect(task).toHaveBeenCalledTimes(1);
  await act(async () => resolve({ ok: true }));
  await screen.findByText('요청 적용 완료');
});
it('failed toggles retain authoritative state and show error', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response('{"ok":false,"error":"변경 실패"}')),
  );
  render(
    <CommandProvider>
      <Toggle label="학습" enabled={true} path="/api/modes" field="learning_enabled" />
    </CommandProvider>,
  );
  fireEvent.click(screen.getByRole('switch'));
  await screen.findByRole('alert');
  expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true');
  expect(screen.getByRole('alert').textContent).toContain('변경 실패');
});
