"""Regression checks for sparse market inputs and the real portfolio ledger."""
from pathlib import Path
from contextlib import closing
from tempfile import TemporaryDirectory
import json
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from stockrl.global_online import Experience, GlobalReplayBuffer, IncrementalMarketCSV, MarketObservation, OnlineGlobalAgent, REWARD_VERSION
from stockrl.global_transformer import GlobalMarketPanel, GlobalMarketTransformer, TransformerConfig
from stockrl.market_training import ContextConditionedTransformer, CONTEXT_FEATURES
from stockrl.multiscale import MULTISCALE_FEATURE_COUNT
from stockrl.paper_account import PaperAccount
from stockrl.state_io import atomic_json

ROOT=Path(__file__).resolve().parents[1]


class Panel:
    window=GlobalMarketPanel.window

    def __init__(self):
        self.dates=np.arange(np.datetime64("2026-09-30T01:00"),np.datetime64("2026-09-30T01:08"),
                             np.timedelta64(1,"m")).astype("datetime64[ns]")
        self.symbols=["TEST.KS","CONTEXT"]
        self.groups={"TEST.KS":("KRX","equity"),"CONTEXT":("FX","currency")}
        self.features=np.zeros((8,2,17),np.float32)
        self.closes=np.asarray([[100+i*2,100] for i in range(8)],dtype=np.float64)
        self.observed=np.ones((8,2),bool)
        self.symbol_ids=np.asarray([0,1]); self.market_ids=np.asarray([0,1]); self.asset_ids=np.asarray([0,1])
        self.market_context=np.zeros((8,len(CONTEXT_FEATURES)),np.float32)

    def multiscale_at(self,index):
        return np.full((2,MULTISCALE_FEATURE_COUNT),index,np.float32)


class FixedPolicy(torch.nn.Module):
    _stockrl_uses_market_context=True

    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(0.0))

    def forward(self,features,*args,portfolio_state=None,account_state=None,multiscale_state=None,
                return_allocation=False):
        b,_,n,_=features.shape
        action=0 if portfolio_state[0,0,0]>0.5 else 2
        logits=torch.full((b,n,3),-10.0)
        logits[:,:,1]=0; logits[:,0,action]=10
        values=torch.zeros((b,n))
        allocation=torch.zeros((b,n+1)); allocation[:,0]=.5; allocation[:,-1]=.5
        return logits,values,allocation


class PipelineTests(unittest.TestCase):
    def test_replay_learning_drains_partial_batches_without_new_quotes(self):
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer()
        for row_id in (1,2):
            row=SimpleNamespace(source="paper_account_symbol",reward_version=REWARD_VERSION,
                features=np.zeros((4,2,17)),portfolio_state=None,account_state=None,
                multiscale_state=None,timestamp=f"T{row_id}")
            agent.replay.items.append(row)
            agent.replay.row_ids[id(row)]=row_id
        agent.min_replay=2; agent.batch_size=2; agent.candidate_replay_passes=2
        agent.candidate_retry_after=0
        for uses,expected in ((1,1),(2,0)):
            agent.metrics={"observation_caught_up":False,"paper_experiences_seen":2,
                           "paper_experiences_since_candidate":0}
            agent.replay._memory_uses={id(row):uses for row in agent.replay.items}
            agent.min_replay=8; agent.batch_size=8
            agent.stop=SimpleNamespace(wait=Mock(side_effect=[False,True]))
            with patch.object(agent,"_train_candidate") as train:
                agent._learner()
                self.assertEqual(train.call_count,expected)
            self.assertEqual(agent.metrics["candidate_eligible_replay_count"],2 if uses==1 else 0)

    @staticmethod
    def experience(timestamp="2026-09-30T01:00:00", features=None, symbol_index=0):
        panel=Panel()
        return Experience(features=panel.features[:4].copy() if features is None else features,
            symbol_ids=panel.symbol_ids,market_ids=panel.market_ids,asset_ids=panel.asset_ids,
            valid_mask=panel.observed[:4].copy(),symbol_index=symbol_index,action=1,reward=.001,
            timestamp=timestamp,source="paper_account_symbol",regime=0,reward_version=REWARD_VERSION,
            market_context=panel.market_context[:4].copy())

    def test_replay_over_4096_and_warning_survives_restart_without_eviction(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(capacity=2,journal_path=path)
            replay.storage_warning_bytes=1
            rows=[self.experience(str(np.datetime64("2026-09-30T01:00:00")+np.timedelta64(i,"s")))
                  for i in range(4100)]
            replay.add_many(rows)
            self.assertEqual(len(replay),4100)
            self.assertTrue(replay.stats()["storage_pressure"])
            restored=GlobalReplayBuffer(capacity=2,journal_path=path)
            self.assertEqual(len(restored),4100)
            self.assertEqual(restored.pending_batch(1)[0].timestamp,rows[0].timestamp)

    def test_shared_rolling_frames_and_checkpoint_ack_are_idempotent(self):
        import sqlite3
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path)
            frames=np.arange(5*2*17,dtype=np.float32).reshape(5,2,17)
            first=self.experience(features=frames[:4]); second=self.experience("2026-09-30T01:01:00",frames[1:])
            replay.add_many([first,second])
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0],5)
            replay=GlobalReplayBuffer(journal_path=path)
            selected=replay.pending_batch(8)
            np.testing.assert_array_equal(selected[1].features,frames[1:])
            ids=replay.row_ids_for(selected)
            replay.acknowledge_training(dict.fromkeys(ids,1))
            self.assertEqual(len(replay),2)
            # Restart before the second pass preserves both remaining rows.
            replay=GlobalReplayBuffer(journal_path=path)
            self.assertEqual(replay.stats()["eligible"],2)
            replay.acknowledge_training(dict.fromkeys(ids,2))
            replay.acknowledge_training(dict.fromkeys(ids,2))
            self.assertEqual(len(replay),0)
            self.assertEqual(replay.stats()["daily"][0]["completed"],2)
            self.assertEqual(replay.stats()["daily"][0]["exposures"],4)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0],0)

    def test_pending_input_survives_weekend_restart_without_csv(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path)
            exp=self.experience(); decision={**vars(exp),"symbol":"TEST.KS","input_symbols":["TEST.KS","CONTEXT"]}
            replay.save_pending([],[decision])
            restored=GlobalReplayBuffer(journal_path=path).load_pending("portfolio")
            np.testing.assert_array_equal(restored[0]["features"],exp.features)
            panel=Panel(); panel.dates+=np.timedelta64(4,"D")
            agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
            agent.replay=GlobalReplayBuffer(journal_path=path)
            agent.paper_account=PaperAccount.in_memory(.001,.0001)
            agent.metrics={}; agent.horizon_kind="seconds"; agent.horizon_amount=60
            self.assertEqual(agent._mature_portfolio(restored,panel,2),[])
            self.assertEqual(len(agent.replay),2)
            self.assertEqual(agent.replay.load_pending("portfolio"),[])

    def test_unexecuted_buy_is_retained_with_zero_trade_reward(self):
        panel=Panel(); agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer(); agent.paper_account=PaperAccount.in_memory(.001,.0001)
        agent.metrics={}; agent.horizon_kind="seconds"; agent.horizon_amount=60
        decision={**vars(self.experience()),"symbol":"TEST.KS","action":2,"fill_expected":False}
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[])
        self.assertEqual(len(agent.replay),2)
        for exp in agent.replay.items:
            self.assertFalse(exp.trade_executed)
            self.assertEqual(exp.reward,0)

    def test_reader_never_trims_unprocessed_market_bars(self):
        import pandas as pd
        reader=IncrementalMarketCSV(ROOT/"unused.csv",retain_timestamps=8)
        frame=pd.DataFrame({"date":pd.date_range("2026-09-30",periods=400,freq="min",tz="UTC"),
                            "symbol":["TEST.KS"]*400})
        self.assertEqual(len(reader._trim(frame)),400)
        reader.processed_through=str(frame.date.iloc[200])
        trimmed=reader._trim(frame)
        self.assertEqual(len(trimmed),327) # 128 preceding context + all 199 new bars.
        self.assertEqual(trimmed.date.iloc[-1],frame.date.iloc[-1])

    def test_feed_compaction_preserves_unobserved_bars_and_input_history(self):
        import pandas as pd
        from stockrl.live_feed import AppendOnlyMarketCSV
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"market.csv"
            (path.parent/"agent").mkdir()
            dates=pd.date_range("2026-09-30",periods=800,freq="min",tz="UTC")
            atomic_json({"last_timestamp":dates[200].isoformat()},path.parent/"agent"/"live_cursor.json")
            writer=AppendOnlyMarketCSV(path); writer.COMPACT_CSV_BYTES=1; writer._next_csv_compaction=1
            try:
                writer.append([{"date":stamp.isoformat(),"symbol":"TEST.KS","market":"KRX",
                    "asset_class":"equity","close":100} for stamp in dates])
            finally:
                writer.close()
            stored=pd.read_csv(path)
            self.assertEqual(len(stored),727)
            self.assertEqual(pd.to_datetime(stored.date,utc=True).iloc[0],dates[73])
            self.assertEqual(pd.to_datetime(stored.date,utc=True).iloc[-1],dates[799])

    def test_shared_forward_preserves_every_experience_gradient(self):
        torch.manual_seed(9)
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
                              n_markets=4,n_asset_types=4,max_seq_len=8)
        model=GlobalMarketTransformer(cfg)
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device=torch.device("cpu"); agent.metrics={"nonfinite_updates":0}
        agent.champion=model; agent.window=4
        examples=[self.experience(symbol_index=i) for i in (0,1)]
        for exp in examples: exp.behavior_log_prob=-1.0
        for exp in examples:
            args,*_=agent._pack([exp]); logits,values=model(*args)
            (agent._experience_loss(logits,values,None,exp)/2).backward()
        expected={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        args,*_=agent._pack([examples[0]]); logits,values=model(*args)
        torch.stack([agent._experience_loss(logits,values,None,e) for e in examples]).mean().backward()
        for name,p in model.named_parameters():
            if name in expected: torch.testing.assert_close(p.grad,expected[name],atol=1e-6,rtol=1e-5)

    def test_reward_keeps_symbol_index_of_original_input(self):
        panel=Panel(); panel.symbols.reverse()
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer(); agent.paper_account=PaperAccount.in_memory(.001,.0001)
        agent.metrics={}; agent.max_pending_age_seconds=86400
        agent.horizon_kind="seconds"; agent.horizon_amount=60
        decision={"timestamp":str(panel.dates[0]),"symbol":"TEST.KS","symbol_index":0,
            "input_symbols":["TEST.KS","CONTEXT"],"action":1,"reward_version":REWARD_VERSION,
            "features":panel.features[:4].copy(),"symbol_ids":panel.symbol_ids,
            "market_ids":panel.market_ids,"asset_ids":panel.asset_ids,
            "valid_mask":panel.observed[:4].copy(),"equity_before":2.0}
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[])
        self.assertEqual(len(agent.replay.items),2)
        self.assertEqual([x.symbol_index for x in agent.replay.items],[0,0])

    def test_policy_does_not_change_with_universe_size(self):
        logits=np.asarray([[.2,.1,.3]])
        one=OnlineGlobalAgent._account_action_probabilities(logits)
        many=OnlineGlobalAgent._account_action_probabilities(np.repeat(logits,170,axis=0))
        np.testing.assert_allclose(one[0],many[0])
        self.assertEqual(OnlineGlobalAgent._deterministic_actions(many),[2]*170)

    def test_live_champion_and_candidate_share_exploration_and_random_draws(self):
        logits=np.asarray([[.2,.1,.3],[-.1,.4,.2],[.8,.1,-.2]])
        portfolio_state=np.zeros((3,4),dtype=np.float32)
        uniforms=[.05,.48,.93]
        expected_probabilities=OnlineGlobalAgent._account_action_probabilities(
            logits,portfolio_state,explore=True)
        expected_actions=OnlineGlobalAgent._sample_actions(expected_probabilities,uniforms)
        champion_probabilities,champion_actions=OnlineGlobalAgent._paper_policy_actions(
            logits,portfolio_state,uniforms)
        candidate_probabilities,candidate_actions=OnlineGlobalAgent._paper_policy_actions(
            logits,portfolio_state,uniforms)
        np.testing.assert_allclose(champion_probabilities,expected_probabilities)
        np.testing.assert_allclose(candidate_probabilities,expected_probabilities)
        self.assertEqual(champion_actions,expected_actions)
        self.assertEqual(candidate_actions,expected_actions)

    def test_pending_ack_accepts_both_call_styles(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            replay=GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3")
            first={"timestamp":"T1","symbol":"A"}; second={"timestamp":"T2","symbol":"A"}
            replay.save_pending([first,second],[])
            replay.acknowledge_pending("regular",first)
            replay.acknowledge_pending("regular","T2|A")
            self.assertEqual(replay.load_pending("regular"),[])

    def test_sparse_symbols_use_last_actual_observation(self):
        x=torch.arange(1*4*3*2).reshape(1,4,3,2).float()
        valid=torch.tensor([[[True,True,False],[True,False,False],[False,True,False],[False,False,False]]])
        result=GlobalMarketTransformer.last_observed_state(x,valid)
        torch.testing.assert_close(result[0,0],x[0,1,0])
        torch.testing.assert_close(result[0,1],x[0,2,1])
        torch.testing.assert_close(result[0,2],torch.zeros(2))

    def test_checkpointed_training_and_inference_match(self):
        torch.manual_seed(3)
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
                              n_markets=4,n_asset_types=4,max_seq_len=8)
        model=ContextConditionedTransformer(GlobalMarketTransformer(cfg))
        x=torch.randn(1,8,3,17); ids=torch.tensor([[0,1,2]])
        valid=torch.ones(1,8,3,dtype=torch.bool); valid[0,-2:,1]=False; valid[:,:,2]=False
        context=torch.zeros(1,8,len(CONTEXT_FEATURES))
        model.activation_checkpointing=False; model.eval()
        expected=model(x,ids,ids,ids,valid,context)
        model.activation_checkpointing=True; model.train()
        actual=model(x,ids,ids,ids,valid,context)
        for a,b in zip(expected,actual): torch.testing.assert_close(a,b)
        actual[0].sum().backward()
        self.assertTrue(torch.isfinite(model.backbone.input_proj.weight.grad).all())

    def test_validation_snapshot_has_no_live_array_aliases(self):
        panel=Panel(); frozen=MarketObservation(panel,5,4)
        expected=frozen.features.copy()
        panel.features[:]=99
        np.testing.assert_array_equal(frozen.features,expected)
        self.assertEqual(len(frozen.dates),4)
        np.testing.assert_array_equal(frozen.multiscale_at(3),np.full((2,MULTISCALE_FEATURE_COUNT),5))

    def test_validation_counts_distinct_future_minutes(self):
        panel=Panel(); panel.dates[2]=panel.dates[1]+np.timedelta64(15,"s")
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.validation_active=True; agent.validation_candidate=object(); agent.window=4
        agent.validation_start_after=str(panel.dates[0]); agent.validation_generation=1
        agent.validation_last_enqueued_minute=None; agent.validation_queue=queue.Queue(maxsize=32)
        agent.metrics={}
        for i in (0,1,2,3): agent._collect_candidate_validation(panel,i)
        self.assertEqual(agent.validation_queue.qsize(),2)
        self.assertEqual(agent.metrics["candidate_validation_queue_max_depth"],2)

    def test_backtest_uses_cash_holdings_costs_and_next_bar_fills(self):
        panel=Panel(); agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.window=4; agent.fee=.001; agent.slippage=.0001; agent.champion=FixedPolicy(); agent.horizon="1m"
        out=agent.evaluate_panel(panel,start_index=3)
        self.assertGreater(out["trade_count"],0)
        self.assertGreater(out["books"]["KRW"]["fees"],0)
        self.assertGreater(out["books"]["KRW"]["sell_tax"],0)
        expected=sum(book["equity"]/book["initial_cash"] for book in out["books"].values())/2-1
        self.assertAlmostEqual(out["net_return"],expected)
        self.assertEqual(out["books"]["USD"]["equity"],10_000)
        self.assertEqual(out["evaluation_mode"],"cash_only_sequential_paper_account")

    def test_account_can_add_and_partially_reduce_position(self):
        account=PaperAccount.in_memory(.001,.0001)
        account._fill("TEST.KS","KRW","BUY",100,0,10000,"T1")
        initial=account.state["books"]["KRW"]["positions"]["TEST.KS"]["quantity"]
        account._fill("TEST.KS","KRW","BUY",100,0,5000,"T2")
        added=account.state["books"]["KRW"]["positions"]["TEST.KS"]["quantity"]
        self.assertGreater(added,initial)
        account._fill("TEST.KS","KRW","SELL",110,0,10,"T3")
        self.assertEqual(account.state["books"]["KRW"]["positions"]["TEST.KS"]["quantity"],added-10)

    def test_atomic_state_recovers_from_brief_file_lock(self):
        import os
        replace=os.replace; attempts=[]
        def transient(*args):
            attempts.append(1)
            if len(attempts)<3: raise PermissionError("temporary sharing violation")
            return replace(*args)
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"account.json"
            with patch("stockrl.state_io.os.replace",side_effect=transient): atomic_json({"ok":True},path)
            self.assertEqual(json.loads(path.read_text()),{"ok":True})
            self.assertFalse(path.with_name("account.json.tmp").exists())


if __name__=="__main__": unittest.main()
