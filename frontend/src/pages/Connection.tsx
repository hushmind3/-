import { useEffect, useState } from 'react';
import { api, endpoints } from '../api';
import { usePolling } from '../hooks';
import { obj, str, time } from '../data';
import {
  ActionButton,
  Badge,
  Buttons,
  Card,
  DataPanel,
  ErrorState,
  Loading,
  StatCard,
  Stats,
} from '../components/ui';
export function Connection() {
  const q = usePolling('/api/provider'),
    p = q.data;
  const [environment, setEnvironment] = useState('real'),
    [account, setAccount] = useState(''),
    [appKey, setAppKey] = useState(''),
    [secret, setSecret] = useState(''),
    [ip, setIp] = useState('');
  const [environmentEdited, setEnvironmentEdited] = useState(false);
  useEffect(() => {
    if (p?.environment && !environmentEdited) setEnvironment(p.environment);
  }, [p?.environment, environmentEdited]);
  if (!p) return q.error ? <ErrorState message={q.error} retry={q.refresh} /> : <Loading />;
  const test = obj(p.last_test);
  return (
    <>
      {q.error && <ErrorState message={'이전 상태 표시 중 · ' + q.error} retry={q.refresh} />}
      <Stats>
        <StatCard
          title="인증 정보"
          value={p.saved ? '저장됨' : '미등록'}
          badge={
            <Badge tone={test.ok ? 'good' : 'neutral'}>{test.ok ? '인증 확인' : '확인 대기'}</Badge>
          }
          detail={'환경 ' + str(p.environment) + ' · ' + time(test.checked_at || test.timestamp)}
        />
        <StatCard title="공인 IP" value={ip || '미조회'} detail="키움 접속 IP 등록용" />
      </Stats>
      <Card title="키움 연결 설정">
        <div className="form-grid">
          <label>
            접속 환경
            <select
              value={environment}
              onChange={(e) => {
                setEnvironmentEdited(true);
                setEnvironment(e.target.value);
              }}
            >
              <option value="real">실제 시세</option>
              <option value="paper">모의 시세</option>
            </select>
          </label>
          <label>
            계좌 번호
            <input
              value={account}
              onChange={(e) => setAccount(e.target.value)}
              autoComplete="off"
            />
          </label>
          <label>
            App Key
            <input
              type="password"
              value={appKey}
              onChange={(e) => setAppKey(e.target.value)}
              autoComplete="new-password"
            />
          </label>
          <label>
            Secret
            <input
              type="password"
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              autoComplete="new-password"
            />
          </label>
        </div>
        <Buttons>
          <ActionButton
            label="인증 정보 저장 · 연결"
            disabled={!appKey || !secret}
            task={async () => {
              const result = await api.command(endpoints.connect, {
                environment,
                account,
                app_key: appKey,
                secret,
              });
              setAppKey('');
              setSecret('');
              setAccount('');
              setEnvironmentEdited(false);
              return result;
            }}
          />
          <ActionButton
            label="저장된 인증으로 재확인"
            path={endpoints.testProvider}
            body={{ provider: 'kiwoom', environment }}
            disabled={!p.saved}
          />
          <ActionButton
            label="공인 IP 조회"
            task={async () => {
              const r = await api.publicIp();
              setIp(str(r.ip || r.public_ip));
              return r;
            }}
          />
        </Buttons>
        <p className="muted">시세 연결 설정입니다. 실제 주문 허용 상태는 바뀌지 않습니다.</p>
      </Card>
      <DataPanel title="인증 · 소켓 · 최근 시세 수신 상세" data={p} />
    </>
  );
}
