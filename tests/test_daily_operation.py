"""Daily competition and live universe migration regression checks."""
import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

import torch

from stockrl.instrument_ids import extend_live_symbols
from stockrl.global_online import GlobalReplayBuffer, OnlineGlobalAgent, REWARD_VERSION, MarketObservation, Experience
from stockrl.global_transformer import TransformerConfig, GlobalMarketTransformer
from stockrl.market_training import ContextConditionedTransformer
from stockrl.paper_account import PaperAccount
from test_online_pipeline import Panel, FixedPolicy
from stockrl.daily_encoder import DailyHistoryEncoder
from unittest.mock import patch, MagicMock
import queue
import threading

ROOT=Path(__file__).resolve().parents[1]


class DailyOperationChecks(unittest.TestCase):
    def test_profit_reward_increases_action_and_loss_reward_decreases_it(self):
        panel=Panel();agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device=torch.device("cpu");agent.metrics={"nonfinite_updates":0}
        for reward in (-.0001,.0001):
            with self.subTest(reward=reward):
                experience=Experience(panel.features,panel.symbol_ids,panel.market_ids,
                    panel.asset_ids,panel.observed,0,2,reward,str(panel.dates[0]),
                    behavior_log_prob=float(np.log(1/3)),trade_executed=True)
                logits=torch.zeros(1,2,3,requires_grad=True)
                values=torch.zeros(1,2,requires_grad=True)
                loss=agent._experience_loss(logits,values,None,experience)
                loss.backward()
                # Gradient descent subtracts this gradient: negative profit
                # must lower the chosen BUY logit, positive profit raise it.
                self.assertLess(float(logits.grad[0,0,2])*reward,0)

    def test_replay_size_tolerates_vanishing_sqlite_sidecar(self):
        from types import SimpleNamespace
        replay=GlobalReplayBuffer.__new__(GlobalReplayBuffer)
        replay.journal_path=ROOT/"size-check.sqlite3"
        def size(path,*args,**kwargs):
            if str(path).endswith("-wal"):
                raise FileNotFoundError("SQLite removed the WAL")
            return SimpleNamespace(st_size=100 if path==replay.journal_path else 50)
        with patch.object(Path,"stat",autospec=True,side_effect=size):
            self.assertEqual(replay.disk_bytes(),150)

    def test_learner_retries_statistics_error_and_resumes_training(self):
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.stop=threading.Event();agent.metrics={};agent.replay=MagicMock()
        agent.candidate_replay_passes=1;agent.dual_learning_enabled=False
        agent.candidate_retry_after=0
        agent.replay.stats.side_effect=[FileNotFoundError("WAL disappeared"),
            {"eligible":1,"untrained":1,"model_remaining":{"candidate":1,"champion":0}}]
        agent._train_candidate=MagicMock(side_effect=agent.stop.set)
        with patch.object(agent.stop,"wait",side_effect=lambda timeout:agent.stop.is_set()):
            agent._learner()
        self.assertEqual(agent.replay.stats.call_count,2)
        agent._train_candidate.assert_called_once()
        self.assertEqual(agent.metrics["learner_statistics_errors"],1)
        self.assertIsNone(agent.metrics["learner_statistics_error"])

    def test_candidate_commits_padded_observations_without_trading_padding(self):
        panel=Panel();panel.symbols[1]="__PAD__0__NASDAQ|EQUITY|A"
        panel.groups.pop("CONTEXT");panel.observed[:,1]=False;panel.closes[:,1]=np.nan
        with TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory)
            agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
            agent.replay=GlobalReplayBuffer(journal_path=root/"replay.sqlite3",dual_learning=True)
            for i in (1,2):agent.replay.enqueue_market_observation(MarketObservation(panel,i,8),True,(.9,.9))
            agent.window=8;agent.stop=threading.Event();agent.stop.set()
            agent.metrics={};agent.device=torch.device("cpu")
            agent.champion=FixedPolicy();agent.candidate_live_model=FixedPolicy()
            agent.candidate_live_model_version=3
            agent.candidate_live_model_lock=threading.Lock();agent.candidate_live_inference_lock=threading.Lock()
            agent.candidate_live_account=PaperAccount.in_memory(.001,.0001)
            agent.candidate_portfolio_pending=[];agent.candidate_live_state_path=root/"observer.json"
            agent._mature_portfolio=lambda pending,*args,**kwargs:pending
            agent._candidate_live_worker()
            self.assertEqual(agent.replay.market_observation_stats()["pending"],0)
            self.assertEqual(agent.metrics.get("candidate_live_inference_count"),2)
            self.assertIsNone(agent.metrics.get("candidate_live_error"))
            pending=agent.replay.load_pending("candidate_portfolio")
            self.assertEqual(len(pending),2)
            self.assertTrue(all(row["symbol"]=="TEST.KS" for row in pending))
            self.assertGreater(agent.candidate_live_account.state["books"]["KRW"]["trade_count"],0)

    def test_clone_preserves_weights_dtypes_outputs_and_has_independent_storage(self):
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
            n_markets=4,n_asset_types=4,max_seq_len=8)
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent);agent.cfg=cfg
        for contextual in (False,True):
            with self.subTest(contextual=contextual):
                backbone=GlobalMarketTransformer(cfg).half()
                source=ContextConditionedTransformer(backbone) if contextual else backbone
                if contextual:
                    source._stockrl_uses_market_context=True
                    source._stockrl_symbol_map={"KR|equity|TEST.KS":0}
                source.eval()
                random_state=torch.get_rng_state().clone()
                clone=agent._new_model_like(source,torch.device("cpu"))
                self.assertTrue(torch.equal(random_state,torch.get_rng_state()))
                clone.load_state_dict(source.state_dict());clone.eval()
                for (name,expected),(other,actual) in zip(source.named_parameters(),clone.named_parameters()):
                    self.assertEqual((name,expected.dtype),(other,actual.dtype))
                    self.assertNotEqual(expected.data_ptr(),actual.data_ptr())
                    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
                args=[torch.randn(1,4,2,17).half(),torch.tensor([[0,1]]),
                    torch.tensor([[0,0]]),torch.tensor([[0,0]]),torch.ones(1,4,2,dtype=torch.bool)]
                if contextual:args.append(torch.randn(1,4,16))
                with torch.inference_mode():
                    for expected,actual in zip(source(*args),clone(*args)):
                        torch.testing.assert_close(expected,actual,atol=0,rtol=0)

    def test_observer_publish_reuses_storage_and_invalidates_old_daily_memory(self):
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
            n_markets=4,n_asset_types=4,max_seq_len=8)
        source=ContextConditionedTransformer(GlobalMarketTransformer(cfg))
        source._stockrl_uses_market_context=True;source._stockrl_symbol_map={}
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent);agent.cfg=cfg
        agent.metrics={};agent.candidate_live_model=None
        agent.candidate_live_model_lock=threading.Lock()
        agent._publish_candidate_observer(source,1)
        observer=agent.candidate_live_model;pointer=next(observer.parameters()).data_ptr()
        observer.daily_history_encoder.cache=("old",torch.ones(1))
        with torch.no_grad():next(source.parameters()).add_(.05)
        agent._publish_candidate_observer(source,2)
        self.assertIs(observer,agent.candidate_live_model)
        self.assertEqual(pointer,next(observer.parameters()).data_ptr())
        self.assertEqual(agent.candidate_live_model_version,2)
        self.assertTrue(agent.metrics["candidate_observer_storage_reused"])
        self.assertIsNone(observer.daily_history_encoder.cache)
        self.assertFalse(any(p.requires_grad for p in observer.parameters()))
        torch.testing.assert_close(next(observer.parameters()),next(source.parameters()),atol=0,rtol=0)

    def test_reset_preserves_original_account_when_required_observations_are_pending(self):
        from stockrl.web_app import Supervisor
        with TemporaryDirectory(dir=ROOT) as directory:
            profile=Path(directory);state=profile/"agent";state.mkdir()
            account=PaperAccount(state/"paper_account.json",.001,.0001)
            account.save();before=(state/"paper_account.json").read_bytes()
            replay=GlobalReplayBuffer(journal_path=state/"replay.sqlite3",dual_learning=True)
            replay.enqueue_market_observation(MarketObservation(Panel(),1,8),True,(.2,.7))
            supervisor=Supervisor.__new__(Supervisor)
            supervisor.lock=threading.RLock();supervisor.account_reset_lock=threading.Lock()
            supervisor.stopping=False;supervisor.run_requested=False
            supervisor.profile=profile;supervisor.runtime=profile;supervisor.mode="live"
            supervisor.horizon="1m";supervisor.fee=.001
            result=supervisor.reset_paper_accounts()
            self.assertIn("pending",result["error"])
            self.assertEqual((state/"paper_account.json").read_bytes(),before)
            self.assertEqual(replay.market_observation_stats()["pending"],1)

    def test_candidate_health_does_not_hide_missing_compulsory_observations(self):
        from stockrl.web_app import _candidate_progress_health
        feed={"latest_feed_timestamp_utc":"2026-10-01T01:00:00+00:00"}
        metrics={"candidate_live_last_timestamp":"2026-10-01T00:50:00+00:00",
            "shared_observation":{"pending":10}}
        health=_candidate_progress_health(feed,metrics,True)
        self.assertEqual((health["status"],health["lag_seconds"],health["pending"]),("stale",600,10))
        metrics["candidate_live_last_timestamp"]="2026-10-01T00:59:00+00:00"
        self.assertEqual(_candidate_progress_health(feed,metrics,True)["status"],"healthy")
        metrics["candidate_live_error"]="input failed"
        self.assertEqual(_candidate_progress_health(feed,metrics,True)["status"],"error")

    def test_daily_store_keeps_five_years_and_excludes_unfinished_days(self):
        from stockrl.multiscale import DailyBarStore, MultiscaleFeatures, _load_daily
        import pandas as pd
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"daily.sqlite3";store=DailyBarStore(path)
            days=pd.bdate_range("2018-01-01",periods=1600)
            store.upsert([dict(symbol="TEST.KS",date=str(day),open=100+i,
                high=101+i,low=99+i,close=100+i,volume=1000+i) for i,day in enumerate(days)])
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0],1500)
            # The latest row exists in storage but its day has not finished yet.
            stamp=np.datetime64(str(days[-1].date())+"T12:00:00","ns")
            frame=pd.DataFrame([dict(symbol="TEST.KS",date=str(stamp),open=1,high=1,low=1,close=1,volume=1)])
            features=MultiscaleFeatures(frame,["TEST.KS"],path,stamp)
            history=features.daily_at(stamp)
            self.assertEqual(history.shape,(1,1300,6))
            self.assertEqual(float(history[...,5].sum()),1300)
            self.assertLess(features.bars[("TEST.KS","1d")][0][-2],int(stamp.astype(np.int64)))
            expected_last_close=100+1598
            last_return=100*(expected_last_close/(100+1597)-1)
            self.assertAlmostEqual(float(history[0,-1,3]),last_return,places=3)
            earlier=int(days[-20].value)
            self.assertTrue(all(row[0]<=earlier for row in _load_daily(path,{"TEST.KS"},earlier)["TEST.KS"]))
            store.close()

    def test_sor_subscription_is_one_identity_per_stock_and_trade_direction_is_retained(self):
        from stockrl.kiwoom_stream import KiwoomRealtimeStream
        instruments=[{"symbol":f"{i:06d}.KS","market":"KRX","asset_class":"equity"} for i in range(105)]
        with patch("stockrl.kiwoom_stream.read_settings",return_value={"environment":"real"}):
            stream=KiwoomRealtimeStream(ROOT,instruments,threading.Event(),queue.Queue())
        self.assertEqual(len(stream.subscription_codes),105)
        self.assertTrue(all(code.endswith("_AL") for code in stream.subscription_codes))
        for volume in ("+5","-3"):
            stream._on_message({"trnm":"REAL","data":[{"type":"0B","item":"000001_AL",
                "values":{"20":"090000","10":"100","15":volume,"27":"101","28":"99","9081":"2"}}]})
        bar=stream.bars["000001.KS"]
        self.assertEqual((bar["volume"],bar["buy_volume"],bar["sell_volume"]),(8,5,3))
        self.assertEqual(stream.stats["nxt_ticks"],2)

    def test_observation_fifo_survives_restart_and_cleanup_without_duplicates(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            panel=Panel()
            for index in (2,1,2):
                replay.enqueue_market_observation(MarketObservation(panel,index,8),True,(.2,.7))
            replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            first,data=replay.next_market_observation()
            self.assertEqual(first,str(panel.dates[1]))
            self.assertEqual(replay.market_observation_stats()["pending"],2)
            with closing(replay._connect()) as db,db:
                replay._collect_unused_windows(db)
            np.testing.assert_array_equal(replay.next_market_observation()[1]["features"],panel.features[:2])
            replay.commit_observer(first,{"last_timestamp":first,"books":{}},[])
            self.assertEqual(replay.observer_account()["last_timestamp"],first)
            self.assertEqual(replay.next_market_observation()[0],str(panel.dates[2]))
            self.assertEqual(replay.market_observation_stats()["candidate_completed"],1)

    def test_two_accounts_same_action_are_distinct_experiences(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            replay=GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3",dual_learning=True)
            panel=Panel()
            base=dict(features=panel.features[:2],symbol_ids=panel.symbol_ids,
                market_ids=panel.market_ids,asset_ids=panel.asset_ids,valid_mask=panel.observed[:2],
                symbol_index=0,action=2,reward=.1,timestamp=str(panel.dates[1]),
                source="paper_account_symbol",reward_version=REWARD_VERSION)
            a=Experience(**base,origin_model="champion");b=Experience(**base,origin_model="candidate")
            replay.add_many([a,b,a,b])
            self.assertEqual(len(replay),2)
            self.assertEqual(replay.market_observation_stats()["experience_origins"],{"champion":1,"candidate":1})

    def test_daily_encoder_reads_all_completed_days_and_invalidates_cache_after_learning(self):
        encoder=DailyHistoryEncoder().eval()
        history=torch.randn(1,2,1300,6);history[...,5]=1
        with torch.no_grad():
            expected=encoder(history);encoder(history)
            self.assertEqual(encoder.cache_hits,1)
            encoder.project[0].weight.add_(.01)
            actual=encoder(history)
            self.assertEqual(encoder.cache_builds,2)
            self.assertFalse(torch.equal(actual,expected))
        encoder.train();history.requires_grad_(True)
        encoder(history).sum().backward()
        self.assertTrue(torch.isfinite(history.grad).all())
        self.assertGreater(float(history.grad[0,0,0,:5].abs().sum()),0)
        self.assertGreater(float(history.grad[0,0,-1,:5].abs().sum()),0)
    def test_universe_extension_never_reuses_or_remaps_existing_ids(self):
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=2,n_markets=4,n_asset_types=4,max_seq_len=8)
        model=ContextConditionedTransformer(GlobalMarketTransformer(cfg))
        model._stockrl_symbol_map={"KR|equity|OLD.KS":0,"NASDAQ|EQUITY|USOLD":1}
        before=model.backbone.symbol_embedding.weight.detach().clone()
        items=[{"symbol":"USOLD","market":"US","asset_class":"equity"},
               {"symbol":"NEW.KS","market":"KRX","asset_class":"equity"}]
        expanded=extend_live_symbols(model,cfg,items)
        self.assertEqual(expanded.max_symbols,3)
        self.assertEqual(model._stockrl_symbol_map,{"KR|equity|OLD.KS":0,"NASDAQ|EQUITY|USOLD":1,"KR|equity|NEW.KS":2})
        torch.testing.assert_close(model.backbone.symbol_embedding.weight[:2],before,atol=0,rtol=0)
        torch.testing.assert_close(model.backbone.symbol_embedding.weight[2],before[0])
        self.assertEqual(extend_live_symbols(model,expanded,items).max_symbols,3)

    def test_account_reset_terminal_reward_uses_old_equity_not_new_seed(self):
        panel=Panel(); agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer();agent.paper_account=PaperAccount.in_memory(.001,.0001)
        agent.metrics={};agent.max_pending_age_seconds=86400
        agent.horizon_kind="seconds";agent.horizon_amount=60
        decision={"timestamp":str(panel.dates[0]),"symbol":"TEST.KS","symbol_index":0,
            "input_symbols":panel.symbols,"action":1,"reward_version":REWARD_VERSION,
            "features":panel.features[:4].copy(),"symbol_ids":panel.symbol_ids,
            "market_ids":panel.market_ids,"asset_ids":panel.asset_ids,
            "valid_mask":panel.observed[:4].copy(),"equity_before":1.99,
            "reset_terminal":True,"reset_equity":1.985,"reset_symbol_net_pnl":0.0}
        self.assertEqual(agent._mature_portfolio([decision],panel,0),[])
        rewards={e.source:(e.portfolio_reward if e.portfolio_transition else e.reward) for e in agent.replay.items}
        self.assertEqual(rewards["paper_account_symbol"],0)
        self.assertAlmostEqual(rewards["paper_account_portfolio"],-.005)


if __name__=="__main__":
    unittest.main()
