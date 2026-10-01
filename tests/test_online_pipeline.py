"""Regression checks for sparse market inputs and the real portfolio ledger."""
from pathlib import Path
from contextlib import closing
from tempfile import TemporaryDirectory
import json
import queue
import unittest
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from stockrl.global_online import Experience, GlobalReplayBuffer, IncrementalMarketCSV, MarketObservation, OnlineGlobalAgent, REWARD_VERSION, save_model
from stockrl.global_transformer import GlobalMarketPanel, GlobalMarketTransformer, TransformerConfig
from stockrl.market_training import ContextConditionedTransformer, CONTEXT_FEATURES
from stockrl.multiscale import MULTISCALE_FEATURE_COUNT
from stockrl.paper_account import PaperAccount
from stockrl.account_diagnostics import summarize_account, summarize_policy
from stockrl.operating_rules import daily_boundary, operating_rules
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
            replay.acknowledge_training(dict.fromkeys(ids,1),passes=2)
            self.assertEqual(len(replay),2)
            # Restart before the second pass preserves both remaining rows.
            replay=GlobalReplayBuffer(journal_path=path)
            self.assertEqual(replay.stats(2)["eligible"],2)
            replay.acknowledge_training(dict.fromkeys(ids,2),passes=2)
            replay.acknowledge_training(dict.fromkeys(ids,2),passes=2)
            self.assertEqual(len(replay),0)
            self.assertEqual(replay.stats()["daily"][0]["completed"],2)
            self.assertEqual(replay.stats()["daily"][0]["exposures"],4)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0],0)

    def test_dual_replay_requires_both_saved_model_passes(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            replay.add(self.experience())
            ids=replay.row_ids_for(replay.pending_batch(8))
            replay.acknowledge_training(dict.fromkeys(ids,2),passes=2,learner="candidate")
            self.assertEqual(len(replay),1)
            self.assertEqual(replay.stats(2)["model_remaining"],{"candidate":0,"champion":1})
            replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            self.assertEqual(replay.stats()["eligible"],1)
            replay.acknowledge_training(dict.fromkeys(ids,1),passes=2,learner="champion")
            self.assertEqual(len(replay),1)
            replay.acknowledge_training(dict.fromkeys(ids,2),passes=2,learner="champion")
            replay.acknowledge_training(dict.fromkeys(ids,2),passes=2,learner="champion")
            self.assertEqual(len(replay),0)
            self.assertEqual(replay.stats()["daily"][0]["completed"],1)
            self.assertEqual(replay.stats()["daily"][0]["exposures"],4)

    def test_both_models_train_restore_and_frozen_pair_competes(self):
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
                              n_markets=4,n_asset_types=4,max_seq_len=8)
        with TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory); model_dir=root/"models"; model_dir.mkdir()
            save_model(model_dir/"champion.pt",GlobalMarketTransformer(cfg),cfg)
            with patch("stockrl.paths.validate_model_dir",return_value=model_dir):
                agent=OnlineGlobalAgent(root/"state",device="cpu",window=4,
                    batch_size=2,updates_per_candidate=1,model_dir=model_dir)
                agent.replay.add_many([self.experience(symbol_index=i) for i in (0,1)])
                original=agent.champion
                original_state={name:p.detach().clone() for name,p in original.state_dict().items()}
                agent._train_champion()
                for name,p in original.state_dict().items(): torch.testing.assert_close(p,original_state[name])
                self.assertIsNot(agent.champion,original)
                self.assertEqual(agent.metrics["last_champion_samples_trained"],2)
                self.assertEqual(len(agent.replay),2)
                # A restart restores the champion's saved training uses once.
                agent=OnlineGlobalAgent(root/"state",device="cpu",window=4,
                    batch_size=2,updates_per_candidate=1,model_dir=model_dir)
                self.assertEqual(agent.replay.stats()["daily"][0]["exposures"],2)
                agent._train_candidate()
                self.assertEqual(len(agent.replay),0)
                frozen={name:p.detach().clone() for name,p in agent.validation_champion.state_dict().items()}
                trial=json.loads(agent.validation_state_path.read_text(encoding="utf-8"))
                self.assertTrue(trial["both_models_frozen"])
                agent.replay.add_many([self.experience("2026-09-30T01:01:00",symbol_index=i) for i in (0,1)])
                agent._train_champion()
                for name,p in agent.validation_champion.state_dict().items(): torch.testing.assert_close(p,frozen[name])
                self.assertEqual(len(agent.replay),2)
                agent._train_candidate()
                self.assertEqual(len(agent.replay),0)
                # Champion may continue learning after the fair trial snapshot.
                # The gate compares the two frozen versions, not today's hashes.
                self.assertNotEqual(agent._sha256_file(agent.champion_path),trial["source_champion_sha256"])
                self.assertFalse(agent._commit_candidate(agent.validation_candidate,-.01,-.02,390,
                    trial["source_champion_sha256"],trial_state=trial))
                self.assertFalse(agent._commit_candidate(agent.validation_candidate,.1,0.0,389,
                    trial["source_champion_sha256"],trial_state=trial))
                self.assertTrue(agent._commit_candidate(agent.validation_candidate,.1,0.0,390,
                    trial["source_champion_sha256"],trial_state=trial))
                self.assertEqual(agent.metrics["promotions"],1)

    def test_single_pass_each_model_survives_restart_and_does_not_repeat(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            replay.add_many([self.experience(),self.experience("2026-09-30T01:01:00")])
            batch=replay.pending_batch(8)
            uses=dict.fromkeys(replay.row_ids_for(batch),1)
            replay.acknowledge_training(uses,learner="candidate")
            replay.acknowledge_training(uses,learner="candidate")
            restored=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            self.assertEqual(restored.pending_batch(8,learner="candidate"),[])
            self.assertEqual(len(restored.pending_batch(8,learner="champion")),2)
            restored.acknowledge_training(uses,learner="champion")
            restored.acknowledge_training(uses,learner="champion")
            self.assertEqual(len(restored),0)
            self.assertEqual(restored.stats()["daily"][0]["exposures"],4)
            self.assertEqual(restored.stats()["daily"][0]["completed"],2)

    def test_pass_target_change_only_consumes_both_checkpoint_confirmed_rows(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            replay=GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3",dual_learning=True)
            replay.add_many([self.experience(),self.experience("2026-09-30T01:01:00")])
            ids=replay.row_ids_for(replay.pending_batch(8,passes=2))
            replay.acknowledge_training(dict.fromkeys(ids,1),passes=2,learner="candidate")
            replay.acknowledge_training({ids[0]:1},passes=2,learner="champion")
            self.assertEqual(replay.finalize_completed(1),1)
            self.assertEqual(replay.finalize_completed(1),0)
            remaining=replay.pending_batch(8,learner="champion")
            self.assertEqual(len(remaining),1)
            self.assertEqual(remaining[0].timestamp,"2026-09-30T01:01:00")
            self.assertEqual(replay.stats()["daily"][0]["completed"],1)
            self.assertEqual(replay.stats()["daily"][0]["exposures"],3)

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


class AccountDiagnosticsTests(unittest.TestCase):
    def test_long_context_uses_completed_history_and_caches_unchanged_windows(self):
        import pandas as pd
        from stockrl.multiscale import DailyBarStore, MultiscaleFeatures, BASE_MULTISCALE_FEATURE_COUNT
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"daily.sqlite3";store=DailyBarStore(path)
            dates=pd.date_range("2024-01-01",periods=600,freq="D",tz="UTC")
            store.upsert([{"date":str(date),"symbol":"TEST.KS","open":100+i,
                "high":101+i,"low":99+i,"close":100+i,"volume":1000+i}
                for i,date in enumerate(dates)])
            store.close()
            frame=pd.DataFrame({"date":[dates[0]],"symbol":["TEST.KS"],
                "open":[100],"high":[101],"low":[99],"close":[100],"volume":[1000]})
            timestamp=np.datetime64("2024-02-15T10:00")
            builder=MultiscaleFeatures(frame,["TEST.KS"],path,timestamp)
            first=builder.at(timestamp)
            # Rows after the point-in-time cutoff cannot supply a full 600-day history.
            self.assertLess(float(first[0,BASE_MULTISCALE_FEATURE_COUNT+5*6+5]),.1)
            self.assertGreater(float(first[0,BASE_MULTISCALE_FEATURE_COUNT+5]),.99)
            cached=len(builder.summary_cache)
            second=builder.at(timestamp+np.timedelta64(1,"m"))
            self.assertEqual(len(builder.summary_cache),cached)
            np.testing.assert_array_equal(first[0,BASE_MULTISCALE_FEATURE_COUNT:BASE_MULTISCALE_FEATURE_COUNT+4],
                                          second[0,BASE_MULTISCALE_FEATURE_COUNT:BASE_MULTISCALE_FEATURE_COUNT+4])

    def test_daily_boundary_is_once_at_seven_kst(self):
        before=datetime(2026,10,1,21,59,tzinfo=timezone.utc)
        after=datetime(2026,10,1,22,0,tzinfo=timezone.utc)
        self.assertEqual(daily_boundary(before),("2026-10-01","2026-10-01T22:00:00+00:00"))
        self.assertEqual(daily_boundary(after),("2026-10-02","2026-10-02T22:00:00+00:00"))

    def test_extended_context_preserves_existing_adapter_weights(self):
        from stockrl.market_training import ContextConditionedTransformer
        from stockrl.global_transformer import load_compatible_state_dict
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,n_markets=4,n_asset_types=4,max_seq_len=8)
        model=ContextConditionedTransformer(GlobalMarketTransformer(cfg))
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        for name in ("multiscale_policy.0.weight","multiscale_value.0.weight"):
            state[name]=state[name][:,:48]
        load_compatible_state_dict(model,state)
        for name in ("multiscale_policy.0.weight","multiscale_value.0.weight"):
            torch.testing.assert_close(model.state_dict()[name][:,:48],state[name])
            self.assertEqual(float(model.state_dict()[name][:,48:].abs().sum()),0)

    def test_fill_statistics_do_not_invent_old_position_age(self):
        account=PaperAccount.in_memory(.001,.0001)
        account._fill("TEST","USD","BUY",100,0,1000,"2026-10-01T00:00:00")
        account._fill("TEST","USD","SELL",102,0,1,"2026-10-01T00:05:00")
        statistics=account.state["books"]["USD"]["trade_statistics"]
        self.assertEqual(statistics["holding_count"],1)
        self.assertEqual(statistics["holding_seconds_sum"],300)
        self.assertEqual(statistics["sell_wins"],1)
        position=account.state["books"]["USD"]["positions"]["TEST"]
        position.pop("opened_timestamp")
        account._fill("TEST","USD","SELL",102,0,1,"2026-10-01T00:10:00")
        self.assertEqual(statistics["holding_count"],1)

    def test_reset_retains_pending_and_ends_its_old_account_episode(self):
        from stockrl.web_app import Supervisor
        with TemporaryDirectory(dir=ROOT) as directory:
            supervisor=Supervisor.__new__(Supervisor)
            supervisor.runtime=Path(directory); supervisor.mode="live"; supervisor.horizon="1m"
            supervisor.profile=Path(directory)/"live"; state=supervisor.profile/"agent";state.mkdir(parents=True)
            supervisor.fee=.001;supervisor.lock=threading.RLock();supervisor.account_reset_lock=threading.Lock()
            supervisor.stopping=False;supervisor.run_requested=False;supervisor._latest_csv_cache={}
            supervisor.daily_cycle_pending=False;supervisor.operating_rules=operating_rules()
            account=PaperAccount(state/"paper_account.json",.001,.0001)
            account._fill("TEST.KS","KRW","BUY",100,0,1000,"2026-09-30T01:00:00")
            account.state["books"]["KRW"]["marks"]["TEST.KS"]=100;account.save()
            replay=GlobalReplayBuffer(journal_path=state/"replay.sqlite3",dual_learning=True)
            replay.save_pending([],[{"timestamp":"2026-09-30T01:00:00","symbol":"TEST.KS","fill_seen":True}])
            self.assertTrue(supervisor.reset_paper_accounts()["ok"])
            pending=replay.load_pending("portfolio")
            self.assertEqual(len(pending),1)
            self.assertTrue(pending[0]["reset_terminal"])
            self.assertNotEqual(pending[0]["reset_equity"],2.0)
            self.assertEqual(PaperAccount(state/"paper_account.json",.001,.0001).state["books"]["KRW"]["positions"],{})
            self.assertEqual(len(supervisor._daily_cycle_status()["history"]),1)

    def test_net_cost_and_holdings_use_one_nonmutating_formula(self):
        account={"last_timestamp":"2026-10-01T00:00:00","pending":{},"fills":[],
            "books":{"USD":{"initial_cash":1000,"cash":890,
                "positions":{"TEST":{"quantity":1,"average_cost":100}},
                "marks":{"TEST":95},"realized_pnl":-10,"fees":8,"slippage":2}}}
        before=json.dumps(account,sort_keys=True)
        book=summarize_account(account)["books"]["USD"]
        self.assertEqual(book["equity"],985)
        self.assertEqual(book["net_pnl"],-15)
        self.assertEqual(book["recorded_pnl_plus_costs"],-5)
        self.assertTrue(book["reconciliation_ok"])
        self.assertAlmostEqual(book["net_return_rate"],-.015)
        self.assertAlmostEqual(book["positions"][0]["weight"],95/985)
        self.assertEqual(json.dumps(account,sort_keys=True),before)

    def test_policy_measures_only_observed_tradable_symbols(self):
        panel=Panel(); account=PaperAccount.in_memory(.001,.0001)
        before=json.dumps(account.state,sort_keys=True)
        result=summarize_policy(panel,0,np.full((2,3),1/3),[0,2],
            [.4,.4,.2],account,set(),str(panel.dates[0]))
        self.assertEqual(result["observed_tradable_symbols"],1)
        self.assertEqual(result["action_counts"],{"SELL":1,"HOLD":0,"BUY":0})
        self.assertEqual(result["sell_without_position"],1)
        self.assertAlmostEqual(result["mean_normalized_entropy"],1)
        self.assertAlmostEqual(result["effective_cash_target"]["KRW"],1/3)
        self.assertIsNone(result["effective_cash_target"]["USD"])
        self.assertEqual(json.dumps(account.state,sort_keys=True),before)

    def test_inconsistent_ledger_is_flagged_not_silently_repaired(self):
        account={"books":{"USD":{"initial_cash":1000,"cash":900}}}
        book=summarize_account(account)["books"]["USD"]
        self.assertFalse(book["reconciliation_ok"])
        self.assertEqual(book["reconciliation_difference"],-100)
        self.assertEqual(account["books"]["USD"]["cash"],900)


if __name__=="__main__": unittest.main()
