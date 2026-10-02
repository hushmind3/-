"""Frozen native stock policies, their original observation rules and symbol masks.

Portfolio policies require their entire training universe at one as-of date.
Missing observations are unavailable, never zero-filled or borrowed from crypto.
"""
import ast
import io
import logging
import time
import cloudpickle
import numpy as np
import pandas as pd
import torch
from torch import nn


INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
              'close_30_sma', 'close_60_sma']


def original_definitions(source, names, namespace):
    """Execute selected native definitions, excluding training/notebook side effects."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
    exec(compile(tree, '<embedded native policy source>', 'exec'), namespace)
    return namespace


def make_policy(spec):
    if spec['kind'] == 'dapo':
        from gymnasium.spaces import Box
        scope = original_definitions(spec['architecture_source'],
            {'mlp', 'Actor', 'MLPCategoricalActor', 'MLPGaussianActor', 'MLPActorCritic'},
            dict(nn=nn, torch=torch, np=np, device=torch.device('cpu'), Box=Box,
                 Normal=torch.distributions.Normal, Categorical=torch.distributions.Categorical))
        return scope['MLPActorCritic'](Box(-np.inf, np.inf, (spec['observation_size'],)),
            Box(-1., 1., (len(spec['universe']),)), hidden_sizes=(512, 512), activation=nn.Tanh)
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.sac.policies import SACPolicy
    cls = SACPolicy if spec['kind'] == 'sac' else ActorCriticPolicy
    return cls(**cloudpickle.loads(spec['constructor']))


def history_frame(records, as_of):
    df = pd.DataFrame(records).copy()
    df = df.rename(columns={c: {'date': 'Date', 'tic': 'Ticker', 'symbol': 'Ticker',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}.get(c, c)
        for c in df.columns})
    if not {'Ticker', 'Date', 'Close'}.issubset(df.columns):
        raise ValueError('stock policy history requires dated, identified stock prices')
    df['Ticker'] = df.Ticker.str.upper()
    df['Date'] = pd.to_datetime(df.Date)
    return df[df.Date <= pd.Timestamp(as_of)].sort_values(['Date', 'Ticker'])


class StockPolicyExpert(nn.Module):
    """A registered native PyTorch policy; construction metadata contains no weights."""
    def __init__(self, policy, entry):
        super().__init__()
        self.models = nn.ModuleList([policy])
        self.entry = entry
        self.requires_grad_(False).eval()

    @classmethod
    def restore(cls, entry, state):
        policy = make_policy(entry['stock_policy'])
        policy.load_state_dict(state, strict=True, assign=True)
        return cls(policy, entry)

    def applicable(self, symbol):
        return symbol.upper() in self.entry['stock_policy']['universe']

    def prepare_input(self, snapshot):
        records = snapshot.get('stock_policy_history')
        if records is None or not any(self.applicable(s) for s in snapshot['symbols']):
            return None
        spec = self.entry['stock_policy']
        df = history_frame(records, snapshot['as_of'])
        account = snapshot.get('policy_account', {})
        cash = float(account.get('cash', 0))
        nav = float(account.get('nav', cash))
        if nav <= 0:
            return None
        positions = account.get('positions', {})
        observations, symbols, prices, dates = [], [], [], []
        if spec['kind'] in ('dapo', 'a2c', 'ppo', 'sac'):
            if not set(spec['indicators']).issubset(df.columns):
                df = self._portfolio_features(df, spec['indicators'])
            latest = df[df.Date == df.Date.max()].set_index('Ticker')
            required = spec['universe']
            if not set(required).issubset(latest.index) or latest.index.duplicated().any():
                return None
            latest = latest.loc[required]
            if not set(spec['indicators']).issubset(latest.columns):
                return None
            prices = latest.Close.to_numpy(float)
            shares = np.asarray([positions.get(s, 0.) for s in required], float)
            obs = [cash, *prices, *shares]
            for col in spec['indicators']:
                obs.extend(latest[col].to_numpy(float))
            if spec['kind'] == 'dapo':
                if not {'llm_sentiment', 'llm_risk'}.issubset(latest.columns):
                    return None
                obs.extend(latest.llm_sentiment); obs.extend(latest.llm_risk)
            observations = [obs]; symbols = required; dates = [str(latest.Date.iloc[0])]
        else:
            for symbol in snapshot['symbols']:
                if not self.applicable(symbol):
                    continue
                frame = df[df.Ticker == symbol.upper()].copy()
                if spec['kind'] == 'msft':
                    frame = self._msft_features(frame)
                    count = spec['observation_size'] // 6
                    # Native env excludes the execution day's row from its observation.
                    frame = frame.dropna(subset=spec['features'])
                    if len(frame) < count + 1:
                        continue
                    obs = frame.iloc[-count-1:-1][spec['features']].to_numpy().reshape(-1)
                else:
                    scope = dict(np=np, pd=pd, List=list, logger=logging.getLogger(__name__))
                    tree = ast.parse(spec['preprocessing_source'])
                    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StockDataProcessor')
                    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                        {'calculate_technical_indicators', 'create_lagged_features'}]
                    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<native stock preprocessing>', 'exec'), scope)
                    processor = object.__new__(scope['StockDataProcessor'])
                    frame = processor.create_lagged_features(processor.calculate_technical_indicators(frame))
                    features = spec['scaler']['features']
                    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
                    if len(frame) < 61:
                        continue
                    values = frame.iloc[-61:-1][features].to_numpy(float)
                    values = (values - np.asarray(spec['scaler']['mean'])) / np.asarray(spec['scaler']['scale'])
                    # The released environment uses scaled Close for portfolio features.
                    native_price = values[-1, features.index('Close')]
                    held = positions.get(symbol, 0.) * native_price
                    initial = spec['initial_balance']
                    portfolio = [cash/initial, held/initial, (cash+held)/initial,
                        (cash+held-initial)/initial, account.get('trades', 0)/100,
                        account.get('costs', 0)/initial, account.get('drawdown', 0), account.get('volatility', 0)]
                    obs = np.r_[values.reshape(-1), portfolio]
                observations.append(obs); symbols.append(symbol)
                prices.append(float(frame.Close.iloc[-1])); dates.append(str(frame.Date.iloc[-1]))
        if not symbols:
            return None
        obs = np.asarray(observations, np.float32)
        if obs.shape[1] != spec['observation_size'] or not np.isfinite(obs).all():
            raise ValueError('native stock observation shape or values invalid')
        return dict(observations=obs.tolist(), symbols=symbols, prices=list(prices),
            as_of=max(dates), account=account, native_features_verified=True,
            feature_schema=spec['schema'], requested_symbols=snapshot['symbols'])

    @staticmethod
    def _portfolio_features(frame, indicators):
        """FinRL FeatureEngineer native StockDataFrame indicators; past rows only."""
        from stockstats import StockDataFrame
        result=[]
        for _, group in frame.groupby('Ticker', sort=False):
            group=group.sort_values('Date').copy()
            native=StockDataFrame.retype(group[['Open','High','Low','Close','Volume']].rename(columns=str.lower).reset_index(drop=True))
            for indicator in indicators:
                group[indicator]=native[indicator].to_numpy()
            result.append(group)
        return pd.concat(result,ignore_index=True)

    @staticmethod
    def _msft_features(frame):
        features = ['log_return', 'sma20', 'sma50', 'rsi14', 'macd', 'volume_change']
        if set(features).issubset(frame.columns):
            return frame
        frame['log_return'] = np.log(frame.Close / frame.Close.shift(1))
        frame['sma20'] = frame.Close.rolling(20).mean()
        frame['sma50'] = frame.Close.rolling(50).mean()
        frame['macd'] = frame.Close.ewm(span=12, adjust=False).mean() - frame.Close.ewm(span=26, adjust=False).mean()
        delta = frame.Close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        frame['rsi14'] = 100 - 100 / (1 + gain/(loss+1e-9))
        frame['volume_change'] = frame.Volume.pct_change()
        return frame

    def forward(self, root, data, device='cpu'):
        spec = self.entry['stock_policy']
        if not data.get('native_features_verified') or data.get('feature_schema') != spec['schema']:
            raise ValueError('native stock observation must use this policy schema')
        if not all(self.applicable(s) for s in data['symbols']):
            raise ValueError('stock policy cannot apply outside its trained universe')
        if spec['kind'] in ('dapo', 'a2c', 'ppo', 'sac') and data['symbols'] != spec['universe']:
            raise ValueError('native portfolio observation requires the exact ordered training universe')
        policy = self.models[0]
        parameters = [(p, p.detach()) for p in policy.parameters()]
        buffers = [(m, k, v) for m in policy.modules() for k, v in m._buffers.items() if v is not None]
        started = time.perf_counter()
        try:
            policy.to(device)
            if device.startswith('cuda'): torch.cuda.synchronize()
            transferred = time.perf_counter()
            obs = torch.as_tensor(data['observations'], dtype=torch.float32, device=device)
            with torch.no_grad():
                if spec['kind'] == 'dapo':
                    dist = policy.pi._distribution(obs)
                    raw = dist.mean; std = dist.stddev
                    action = raw.clamp(-1, 1)
                else:
                    raw = policy._predict(obs, deterministic=True)
                    if spec['kind'] == 'msft':
                        probs = policy.get_distribution(obs).distribution.probs
                        std = None; action = raw
                    elif spec['kind'] == 'sac':
                        std = None; action = raw.clamp(-1, 1)
                    else:
                        dist = policy.get_distribution(obs).distribution
                        std = dist.stddev
                        lo, hi = torch.as_tensor(policy.action_space.low, device=device), torch.as_tensor(policy.action_space.high, device=device)
                        action = torch.maximum(lo, torch.minimum(hi, raw))
                if device.startswith('cuda'): torch.cuda.synchronize()
                calculated = time.perf_counter()
                native = raw.cpu().numpy()
                rows = self._adapt(action.cpu().numpy(), data,
                    probs.cpu().numpy() if spec['kind'] == 'msft' else None,
                    std.cpu().numpy() if std is not None else None,native)
        finally:
            for p, value in parameters: p.data = value
            for m, k, value in buffers: m._buffers[k] = value
        # Native portfolio output is retained intact, common evidence is per symbol.
        requested = set(data.get('requested_symbols', data['symbols']))
        rows = [r for r in rows if r['symbol_id'] in requested]
        return dict(expert=self.entry['id'], as_of=data['as_of'], symbols=[r['symbol_id'] for r in rows],
            native_output=[[r['buy_score'], r['hold_score'], r['sell_score'], r['target_weight'], r['confidence']] for r in rows],
            raw_policy_output=native.tolist(), raw_policy_shape=list(native.shape), common_output=rows,
            output_shape=[len(rows),5],input_shapes={'observation':list(obs.shape)},
            layout='symbol,policy_evidence', units='policy_scores_and_account_weight', horizon=1, sampling_seconds=86400,
            frozen=True, native_features_verified=True, cold_load_seconds=0.,
            gpu_transfer_seconds=transferred-started, forward_seconds=calculated-transferred,
            worker_seconds=time.perf_counter()-started, trained_universe=spec['universe'])

    def _adapt(self, action, data, probabilities, std, raw):
        spec = self.entry['stock_policy']; account = data['account']
        cash, nav = float(account['cash']), float(account['nav'])
        positions = account.get('positions', {}); rows = []
        portfolio = spec['kind'] in ('dapo', 'a2c', 'ppo', 'sac')
        targets = {}
        if portfolio:
            actions = action.reshape(-1)
            quantities = np.asarray([positions.get(s, 0.) for s in data['symbols']], float)
            prices = np.asarray(data['prices'], float)
            deltas = (actions*spec['hmax']).astype(int)
            # Preserve native sell-before-buy order and cash-limited share quantities.
            for i in np.argsort(deltas):
                if deltas[i] >= 0: continue
                sold = min(quantities[i], -deltas[i]); quantities[i] -= sold
                cash += sold*prices[i]*(1-spec['transaction_cost'])
            for i in np.argsort(deltas)[::-1]:
                if deltas[i] <= 0: continue
                bought = min(deltas[i], int(cash/(prices[i]*(1+spec['transaction_cost']))))
                quantities[i] += bought; cash -= bought*prices[i]*(1+spec['transaction_cost'])
            targets = dict(zip(data['symbols'], quantities*prices/nav))
        for i, symbol in enumerate(data['symbols']):
            price = data['prices'][i]; held = positions.get(symbol, 0.)*price/nav
            if spec['kind'] == 'msft':
                sell, hold, buy = probabilities[i].tolist()
                target = [0., held, held + account['cash']/nav/(1+spec['transaction_cost'])][int(action[i])]
                confidence = max(sell, hold, buy)
            elif spec['kind'] == 'adilbai':
                kind, fraction = int(action[i, 0]), float(action[i, 1])
                mu = float(raw[i, 0]); sigma = max(float(std[i, 0]), 1e-9)
                # Action type is truncated (not rounded) by the released environment.
                normal = torch.distributions.Normal(mu, sigma)
                hold = float(normal.cdf(torch.tensor(1.)))
                buy = float(normal.cdf(torch.tensor(2.))) - hold; sell = 1-hold-buy
                quantity=positions.get(symbol,0.)
                if kind == 1:
                    bought=int(account['cash']/price*fraction)
                    if bought*price*(1+spec['transaction_cost'])>account['cash']:bought=0
                    quantity+=bought
                elif kind == 2:quantity-=int(quantity*fraction)
                target=quantity*price/nav
                confidence = max(buy, hold, sell)
            else:
                a = float(action.reshape(-1)[i]); buy, sell = max(a, 0.), max(-a, 0.)
                hold = 1-abs(a); target = targets[symbol]; confidence = abs(a)
            rows.append(dict(symbol_id=symbol, buy_score=buy, hold_score=hold, sell_score=sell,
                target_weight=float(np.clip(target, 0, 1)), confidence=confidence))
        return rows
