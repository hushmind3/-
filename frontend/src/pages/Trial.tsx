import type { StatusResponse } from '../types';
import { Assembly } from './Assembly';
// Preserve the old hash route; use the actual MoE competition UI and API.
export function Trial({ status: _status }: { status: StatusResponse }) {
  return <Assembly />;
}
