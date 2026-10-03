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
    def test_split_web_assets_and_public_supervisor(self):
        from stockrl.web.resources import ASSET_NAMES, dashboard_asset, ROOT as WEB_ROOT
        from stockrl.web_app import Supervisor
        self.assertEqual(WEB_ROOT, ROOT)
        self.assertEqual(Supervisor.status.__module__,"stockrl.web.status")
        self.assertEqual(Supervisor.set_modes.__module__,"stockrl.web.runtime")
        for name in ASSET_NAMES:
            body,mime=dashboard_asset("/assets/"+name)
            self.assertTrue(body)
            self.assertEqual(mime,"text/css; charset=utf-8" if name.endswith(".css") else "text/javascript; charset=utf-8")
        self.assertIsNone(dashboard_asset("/assets/../web_dashboard.html"))
        self.assertIsNone(dashboard_asset("/assets/missing.js"))

    def test_extracted_contextual_benchmark_preserves_public_entrypoint(self):
        from stockrl.global_online import benchmark_model
        panel=Panel()
        class ContextPolicy(FixedPolicy):
            _stockrl_uses_market_context=True
            def forward(self,*args,**kwargs):
                batch,_,symbols,_=args[0].shape
                return torch.zeros(batch,symbols,3),torch.zeros(batch,symbols)
        result=benchmark_model(ContextPolicy(),panel,window=4,repeats=1,device=torch.device("cpu"))
        self.assertEqual(result["inference_repeats"],1)
        self.assertGreaterEqual(result["inference_seconds_p50"],0)

    def test_context_quotes_do_not_trigger_stock_policy(self):
        panel=Panel()
        panel.observed[2,0]=False
        self.assertFalse(OnlineGlobalAgent._has_tradable_update(panel,2))
        panel.observed[2,0]=True
        self.assertTrue(OnlineGlobalAgent._has_tradable_update(panel,2))
        panel.groups["TEST.KS"]=("CRYPTO","crypto")
        self.assertFalse(OnlineGlobalAgent._has_tradable_update(panel,2))

    def test_controls_are_independent(self):
        from stockrl.web_app import Supervisor
        supervisor=Supervisor.__new__(Supervisor)
        supervisor.lock=threading.RLock()
        supervisor.autonomy_enabled=True;supervisor.observe_enabled=True
        supervisor.learning_enabled=True;supervisor._write_autonomy=MagicMock()
        result=supervisor.set_modes(observe_enabled=False)
        self.assertFalse(result["observe_enabled"])
        self.assertTrue(result["paper_enabled"])
        self.assertTrue(result["learning_enabled"])
        supervisor.set_modes(paper_enabled=False,learning_enabled=False)
        result=supervisor.set_modes(paper_enabled=True)
        self.assertFalse(result["observe_enabled"])
        self.assertFalse(result["learning_enabled"])
        self.assertTrue(result["paper_enabled"])

    def test_context_observer_commits_without_model_or_gpu(self):
        panel=Panel();panel.observed[:,0]=False
        with TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory);agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
            agent._init_model_lifecycle(False)
            agent.state_dir=root;agent.replay=GlobalReplayBuffer(journal_path=root/"replay.sqlite3",dual_learning=True)
            for i in (1,2):agent.replay.enqueue_market_observation(MarketObservation(panel,i,8),True,())
            agent.window=8;agent.stop=MagicMock();agent.stop.is_set.side_effect=[False,False,False,False,True];agent.metrics={}
            agent.candidate_live_model=None;agent.candidate_live_account=PaperAccount.in_memory(.001,.0001)
            agent.candidate_portfolio_pending=[];agent.candidate_live_state_path=root/"observer.json"
            agent.candidate_live_inference_lock=threading.Lock()
            agent._mature_portfolio=lambda pending,*args,**kwargs:pending
            agent._gpu_work=MagicMock(side_effect=AssertionError("must not request GPU"))
            agent._candidate_live_worker()
            self.assertEqual(agent.replay.market_observation_stats()["pending"],0)
            self.assertEqual(agent.metrics["candidate_context_only_updates"],2)
            self.assertEqual(agent.metrics.get("candidate_live_inference_count",0),0)
            agent._gpu_work.assert_not_called()

    def test_after_twenty_learning_priority_returns_at_us_regular_open(self):
        from datetime import datetime,timezone
        from stockrl.gpu_scheduler import FairGpuScheduler
        now=[datetime(2026,10,1,11,0,tzinfo=timezone.utc)]
        scheduler=FairGpuScheduler(preopen_learning=True,clock=lambda:now[0])
        self.assertTrue(scheduler.learning_first())
        self.assertLess(scheduler.priority("champion_learning_step"),scheduler.priority("champion_live"))
        self.assertEqual(scheduler.snapshot()["learning_priority_until_utc"],"2026-10-01T13:30:00+00:00")
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent._init_model_lifecycle(False)
        agent.live_priority_enabled=True;agent.candidate_live_inference_lock=scheduler
        with scheduler.work("candidate_live"):
            self.assertIsNone(agent._live_learning_wait_reason())
        now[0]=datetime(2026,10,1,13,30,tzinfo=timezone.utc)
        self.assertFalse(scheduler.learning_first())
        self.assertLess(scheduler.priority("champion_live"),scheduler.priority("champion_learning_step"))
        now[0]=datetime(2026,12,1,14,29,tzinfo=timezone.utc)
        self.assertTrue(scheduler.learning_first())
        now[0]=datetime(2026,12,1,14,30,tzinfo=timezone.utc)
        self.assertFalse(scheduler.learning_first())

    def test_preopen_learning_overtakes_waiting_inference_without_removing_it(self):
        from datetime import datetime,timezone
        from stockrl.gpu_scheduler import FairGpuScheduler
        scheduler=FairGpuScheduler(preopen_learning=True,
            clock=lambda:datetime(2026,10,1,11,0,tzinfo=timezone.utc))
        order=[];threads=[]
        def work(role):
            with scheduler.work(role): order.append(role)
        with scheduler.work("running_work"):
            for count,role in enumerate(("candidate_live","validation_candidate","champion_learning_step"),1):
                worker=threading.Thread(target=work,args=(role,));worker.start();threads.append(worker)
                with scheduler.condition:
                    self.assertTrue(scheduler.condition.wait_for(lambda:len(scheduler.queue)==count,2))
        for worker in threads:
            worker.join(2);self.assertFalse(worker.is_alive())
        self.assertEqual(order,["champion_learning_step","candidate_live","validation_candidate"])

    def test_replay_learning_waits_only_for_actual_gpu_inference(self):
        from stockrl.gpu_scheduler import FairGpuScheduler
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent._init_model_lifecycle(False)
        agent.live_priority_enabled=True;agent.metrics={"observation_caught_up":False}
        agent.replay=MagicMock();agent.replay.market_observation_stats.return_value={"pending":3}
        agent.candidate_live_inference_lock=FairGpuScheduler()
        # CPU/DB backlog must not leave the GPU idle while replay is ready.
        self.assertIsNone(agent._live_learning_wait_reason())
        agent.replay.market_observation_stats.assert_not_called()
        with agent.candidate_live_inference_lock.work("candidate_live"):
            self.assertIn("candidate_live",agent._live_learning_wait_reason())
        with agent.candidate_live_inference_lock.work("validation_candidate"):
            self.assertIsNone(agent._live_learning_wait_reason())
        agent.stop=threading.Event()
        self.assertTrue(agent._wait_for_live_inference())
        agent.stop.set();self.assertFalse(agent._wait_for_live_inference())

    def test_learning_segments_yield_to_inference_without_losing_accumulated_gradient(self):
        from stockrl.gpu_scheduler import FairGpuScheduler
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent._init_model_lifecycle(False)
        agent.live_priority_enabled=True;agent.metrics={};agent.device=torch.device("cpu")
        agent.candidate_live_inference_lock=FairGpuScheduler()
        parameter=torch.nn.Parameter(torch.tensor(1.0));durations=[];order=[]
        def infer():
            with agent._gpu_work("candidate_live"): order.append("inference")
        with agent._learning_gpu_segment("candidate_learning_window",durations):
            (parameter*2).backward();order.append("first_window")
            worker=threading.Thread(target=infer);worker.start()
            scheduler=agent.candidate_live_inference_lock
            with scheduler.condition:
                self.assertTrue(scheduler.condition.wait_for(lambda:len(scheduler.queue)==1,2))
        with agent._learning_gpu_segment("candidate_learning_window",durations):
            (parameter*3).backward();order.append("second_window")
        worker.join(2);self.assertFalse(worker.is_alive())
        self.assertEqual(order,["first_window","inference","second_window"])
        self.assertEqual(float(parameter.grad),5.0)
        self.assertEqual(len(durations),2)
        self.assertTrue(all(duration>=0 for duration in durations))

    def test_candidate_health_uses_its_committed_cursor_instead_of_old_champion_metrics(self):
        from stockrl.web_app import _candidate_progress_health
        health={"latest_feed_timestamp_utc":"2026-10-01T10:40:00+00:00"}
        metrics={"candidate_live_last_timestamp":"2026-10-01T10:20:00+00:00",
                 "candidate_live_error":"old error","shared_observation":{"pending":0}}
        observer={"last_timestamp":"2026-10-01T10:39:50+00:00","status":"observing"}
        current=_candidate_progress_health(health,metrics,True,observer)
        self.assertEqual(current["lag_seconds"],10)
        self.assertEqual(current["status"],"healthy")

    def test_live_gpu_priority_overtakes_queued_learning_and_retains_live_fifo(self):
        from stockrl.gpu_scheduler import FairGpuScheduler
        scheduler=FairGpuScheduler();order=[];threads=[]
        def work(role):
            with scheduler.work(role): order.append(role)
        with scheduler.work("active_learning_step"):
            for count,role in enumerate(("candidate_learning_step","candidate_live","champion_live"),1):
                thread=threading.Thread(target=work,args=(role,));thread.start();threads.append(thread)
                with scheduler.condition:
                    self.assertTrue(scheduler.condition.wait_for(lambda:len(scheduler.queue)==count,2))
        for thread in threads:
            thread.join(2);self.assertFalse(thread.is_alive())
        self.assertEqual(order,["candidate_live","champion_live","candidate_learning_step"])

    def test_gpu_fifo_waiting_observation_precedes_next_learning_step(self):
        from stockrl.gpu_scheduler import FairGpuScheduler
        scheduler=FairGpuScheduler();order=[]
        entered=threading.Event()
        def observe():
            entered.set()
            with scheduler.work("candidate_live"):
                order.append("candidate_live")
        with scheduler.work("first_learning_step"):
            worker=threading.Thread(target=observe);worker.start()
            self.assertTrue(entered.wait(2))
            with scheduler.condition:
                self.assertTrue(scheduler.condition.wait_for(lambda:len(scheduler.queue)==1,2))
        with scheduler.work("second_learning_step"):
            order.append("second_learning_step")
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(order,["candidate_live","second_learning_step"])
        with self.assertRaises(RuntimeError):
            with scheduler.work("failed_work"):
                raise RuntimeError("test exception")
        with scheduler.work("recovered"):
            self.assertEqual(scheduler.snapshot()["active"],"recovered")
        self.assertIsNone(scheduler.snapshot()["active"])

    def test_git_split_sqlite_restores_complete_pending_work_without_overwriting_live_db(self):
        from stockrl.runtime_backup import restore_sqlite_backup
        import sqlite3,gzip,json,hashlib
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            with closing(sqlite3.connect(":memory:")) as db:
                db.execute("CREATE TABLE pending_records (value TEXT)")
                db.execute("INSERT INTO pending_records VALUES ('unlearned')");db.commit()
                content=db.serialize()
            parts=[]
            for index,start in enumerate(range(0,len(content),2048)):
                name=path.name+".part%03d.gz"%index
                (path.parent/name).write_bytes(gzip.compress(content[start:start+2048]))
                parts.append({"name":name})
            path.with_name(path.name+".restore.json").write_text(json.dumps({
                "parts":parts,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}))
            self.assertTrue(restore_sqlite_backup(path))
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT value FROM pending_records").fetchone()[0],"unlearned")
            before=path.read_bytes()
            self.assertFalse(restore_sqlite_backup(path))
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(path.with_name(path.name+".restore.tmp").exists())

    def test_joint_goal_win_ends_pending_credit_without_future_bootstrap(self):
        from test_online_pipeline import PipelineTests
        panel=Panel();agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer();agent.metrics={}
        agent.paper_account=PaperAccount.in_memory(0,0)
        agent.paper_account.configure_goal()
        before=agent.paper_account.goal_inputs()
        exp=PipelineTests.experience()
        decision={**vars(exp),"symbol":"TEST.KS","action":1,"timestamp":str(panel.dates[0]),
            "entry_price":100,"equity_before":2,"symbol_pnl_before":0,
            "credit_observations_target":60,"credit_seconds":3600,
            "goal_state":np.asarray(before,np.float32),"goal_points_before":{"KRW":0,"USD":0},
            "goal_weight_before":.25,"goal_complete_before":False,
            "goal_episode_id":agent.paper_account.state["episode_id"]}
        for book in agent.paper_account.state["books"].values():
            book["cash"]=book["initial_cash"]*10
        self.assertEqual(agent._mature_portfolio([decision],panel,1),[])
        rows=agent.replay.pending_batch(8)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(e.goal_terminal and e.bootstrap_discount==0 for e in rows))
        self.assertTrue(all(e.goal_reward_points==25 for e in rows))
        self.assertEqual(next(e.portfolio_goal_reward_points for e in rows if e.portfolio_transition),200)
        np.testing.assert_array_equal(rows[0].goal_state,np.asarray(before,np.float32))

    def test_both_origins_train_goal_heads_through_actual_optimizer(self):
        panel=Panel();agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device=torch.device("cpu");agent.window=4;agent.metrics={"nonfinite_updates":0}
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
            n_markets=4,n_asset_types=4,max_seq_len=4)
        for origin in ("champion","candidate"):
            model=ContextConditionedTransformer(GlobalMarketTransformer(cfg))
            model._stockrl_uses_market_context=True;agent.champion=model
            args=panel.window(0,4,include_context=True)
            exp=Experience(args[0][0].numpy(),panel.symbol_ids,panel.market_ids,panel.asset_ids,
                args[4][0].numpy(),0,2,-.001,str(panel.dates[0]),market_context=args[5][0].numpy(),
                goal_state=np.asarray([1,1,10,.9,.9,1],np.float32),goal_reward_points=1,
                behavior_log_prob=float(np.log(1/3)),origin_model=origin)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
            before=model.goal_policy.weight.detach().clone()
            a,p,s,m=agent._pack([exp])
            logits,values,allocation=model(*a,portfolio_state=p,account_state=s,
                return_allocation=True,multiscale_state=m,**agent._saved_goal_kwargs(exp,agent.device))
            loss=agent._experience_loss(logits,values,allocation,exp)
            loss.backward();opt.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(torch.equal(before,model.goal_policy.weight))
            self.assertGreater(float(model.goal_value.weight.grad.abs().sum()),0)

    def test_goal_win_is_once_per_currency_persistent_and_reset_is_explicit(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            account=PaperAccount(Path(directory)/"account.json",0,0)
            initial=account.state["books"]["KRW"]["cash"]
            episode=account.state["episode_id"]
            account.configure_goal()
            self.assertEqual(account.state["books"]["KRW"]["cash"],initial)
            self.assertEqual(account.goal_inputs(),[1,1,10,.9,.9,1])
            account.state["books"]["KRW"]["cash"]=initial*10
            account.observe_goal("T1");account.save()
            loaded=PaperAccount(account.path,0,0);loaded.configure_goal();loaded.observe_goal("T2")
            self.assertEqual(loaded.goal_points(),{"KRW":100,"USD":0})
            self.assertEqual(loaded.goal_summary()["books"]["KRW"]["win"]["timestamp"],"T1")
            self.assertEqual(loaded.reward_points()["KRW"],900)
            loaded.state["books"]["USD"]["cash"]=100_000
            loaded.observe_goal("T3")
            self.assertEqual(loaded.goal_summary()["status"],"WIN")
            loaded.reset()
            self.assertNotEqual(loaded.state["episode_id"],episode)
            self.assertEqual(loaded.goal_points(),{"KRW":0,"USD":0})
            self.assertEqual(loaded.goal_summary()["target_multiple"],10)

    def test_zero_goal_heads_preserve_legacy_output_and_receive_gradients(self):
        from stockrl.global_transformer import load_compatible_state_dict
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
            n_markets=4,n_asset_types=4,max_seq_len=4)
        model=ContextConditionedTransformer(GlobalMarketTransformer(cfg)).eval()
        legacy={k:v for k,v in model.state_dict().items() if not k.startswith("goal_")}
        load_compatible_state_dict(model,legacy)
        args=Panel().window(0,4,include_context=True)
        baseline=model(*args)
        goal=torch.tensor([[1.,1.,10.,.9,.9,1.]])
        result=model(*args,goal_state=goal)
        for a,b in zip(baseline,result):torch.testing.assert_close(a,b,atol=0,rtol=0)
        (result[0].sum()+result[1].sum()).backward()
        self.assertGreater(float(model.goal_policy.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.goal_value.weight.grad.abs().sum()),0)

    def test_goal_credit_survives_replay_restart_and_both_model_ack(self):
        panel=Panel()
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            exp=Experience(panel.features,panel.symbol_ids,panel.market_ids,panel.asset_ids,
                panel.observed,0,2,.001,str(panel.dates[0]),reward_version=REWARD_VERSION,
                source="paper_account_symbol",goal_state=np.asarray([1,1,10,.9,.9,1],np.float32),
                goal_reward_points=2,portfolio_goal_reward_points=100,goal_terminal=True,
                goal_episode_id="episode",origin_model="candidate")
            replay.add_many([exp]);replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            row=replay.pending_batch(8)[0]
            np.testing.assert_array_equal(row.goal_state,exp.goal_state)
            self.assertEqual((row.goal_reward_points,row.portfolio_goal_reward_points,row.goal_terminal),
                (2,100,True))
            self.assertEqual(row.goal_episode_id,"episode")
            ids=replay.row_ids_for([row])
            replay.acknowledge_training(dict.fromkeys(ids,1),learner="champion")
            self.assertEqual(len(replay),1)
            replay.acknowledge_training(dict.fromkeys(ids,1),learner="candidate")
            self.assertEqual(len(replay),0)

    def test_daily_competition_keeps_long_term_accounts_and_pending(self):
        from stockrl.web_app import Supervisor
        with TemporaryDirectory(dir=ROOT) as directory:
            profile=Path(directory);state=profile/"agent";state.mkdir()
            before={}
            for name in ("paper_account.json","candidate_observer_account.json"):
                account=PaperAccount(state/name,0,0);account.configure_goal()
                account.state["books"]["KRW"]["cash"]=9_000_000
                account.save();before[name]=(state/name).read_bytes()
            replay=GlobalReplayBuffer(journal_path=state/"replay.sqlite3",dual_learning=True)
            replay.enqueue_market_observation(MarketObservation(Panel(),1,8),True,(.2,.7))
            supervisor=Supervisor.__new__(Supervisor)
            supervisor.lock=threading.RLock();supervisor.account_reset_lock=threading.Lock()
            supervisor.stopping=False;supervisor.run_requested=False
            supervisor.profile=profile;supervisor.runtime=profile;supervisor.mode="live"
            supervisor.horizon="1m";supervisor.operating_rules={"daily_reset_live_accounts":False,"daily_history_limit":31}
            supervisor._daily_cycle_status=lambda:{"session_key":"day","current_session_key":"day","history":[]}
            result=supervisor.reset_paper_accounts(daily=True,session_key="day")
            self.assertEqual(result["reset"],[])
            for name in before:self.assertEqual((state/name).read_bytes(),before[name])
            self.assertEqual(replay.market_observation_stats()["pending"],1)

    def test_score_changes_are_once_per_timestamp_and_reset_is_not_profit(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            replay=GlobalReplayBuffer(journal_path=path)
            account=PaperAccount.in_memory(.001,.0001)
            episode=account.state["episode_id"]
            replay.record_account_score("champion","T0",episode,account.reward_points())
            for stamp,points,change in [("T1",1,1),("T2",1,0),("T3",2,1),("T4",.5,-1.5)]:
                account.state["books"]["KRW"]["cash"]=10_000_000*(1+points/100)
                actual=replay.record_account_score("champion",stamp,episode,account.reward_points())
                self.assertAlmostEqual(actual["points"]["KRW"],points)
                self.assertAlmostEqual(actual["change"]["KRW"],change)
                self.assertEqual(replay.record_account_score("champion",stamp,episode,account.reward_points()),actual)
            replay=GlobalReplayBuffer(journal_path=path)
            actual=replay.record_account_score("champion","T5",episode,account.reward_points())
            self.assertAlmostEqual(actual["change"]["KRW"],0)
            account.reset()
            actual=replay.record_account_score("champion","T6",account.state["episode_id"],account.reward_points())
            self.assertIsNone(actual["change"])
            self.assertAlmostEqual(actual["points"]["KRW"],0)

    def test_future_gain_can_outweigh_immediate_cost_without_reward_duplication(self):
        panel=Panel();agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device=torch.device("cpu");agent.metrics={"nonfinite_updates":0}
        exp=Experience(panel.features,panel.symbol_ids,panel.market_ids,panel.asset_ids,
            panel.observed,0,2,-.001,str(panel.dates[0]),
            behavior_log_prob=float(np.log(1/3)),bootstrap_discount=1.0,bootstrap_symbol_index=0)
        logits=torch.zeros(1,2,3,requires_grad=True);values=torch.zeros(1,2,requires_grad=True)
        successor=torch.tensor([[.5,0.0]],requires_grad=True)
        agent._experience_loss(logits,values,None,exp,successor).backward()
        self.assertLess(float(logits.grad[0,0,2]),0)
        self.assertIsNone(successor.grad)
        with self.assertRaises(ValueError):agent._experience_loss(logits,values,None,exp)

    def test_real_optimizer_uses_successor_and_shares_its_forward(self):
        panel=Panel();agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device=torch.device("cpu");agent.window=4;agent.metrics={"nonfinite_updates":0}
        cfg=TransformerConfig(d_model=16,n_heads=2,n_layers=2,max_symbols=4,
            n_markets=4,n_asset_types=4,max_seq_len=4)
        model=ContextConditionedTransformer(GlobalMarketTransformer(cfg))
        model._stockrl_uses_market_context=True;agent.champion=model
        args=panel.window(0,4,include_context=True)
        pstate,astate=PaperAccount.in_memory(0,0).model_inputs(panel,0)
        exp=Experience(args[0][0].numpy(),panel.symbol_ids,panel.market_ids,panel.asset_ids,
            args[4][0].numpy(),0,2,.01,str(panel.dates[0]),market_context=args[5][0].numpy(),
            portfolio_state=pstate,account_state=astate,bootstrap_discount=1,
            bootstrap_symbol_index=0,bootstrap_window_key="shared-next",behavior_log_prob=float(np.log(1/3)))
        exp._bootstrap_inputs={name:getattr(exp,name,None) for name in (
            "features","symbol_ids","market_ids","asset_ids","valid_mask","market_context",
            "portfolio_state","account_state","multiscale_state","daily_history")}
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        cache={}
        with patch.object(model,"forward",wraps=model.forward) as forward:
            future=agent._credit_successor_values(model,exp,cache)
            self.assertIs(agent._credit_successor_values(model,exp,cache),future)
            self.assertEqual(forward.call_count,1)
        self.assertTrue(model.training);self.assertFalse(future.requires_grad)
        a,p,s,m=agent._pack([exp])
        logits,values,allocation=model(*a,portfolio_state=p,account_state=s,
            return_allocation=True,multiscale_state=m)
        loss=agent._experience_loss(logits,values,allocation,exp,future)
        loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_account_reset_ends_old_credit_without_new_seed_bootstrap(self):
        from test_online_pipeline import PipelineTests
        panel=Panel();agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer();agent.metrics={"paper_experiences_seen":0}
        agent.paper_account=PaperAccount.in_memory(0,0)
        agent.horizon_kind="bars";agent.horizon_amount=1
        exp=PipelineTests.experience()
        decision={**vars(exp),"symbol":"TEST.KS","action":1,"timestamp":str(panel.dates[0]),
            "entry_price":100,"equity_before":2,"symbol_pnl_before":0,
            "credit_observations_target":60,"reset_terminal":True,
            "reset_equity":1.98,"reset_symbol_net_pnl":-.02}
        self.assertEqual(agent._mature_portfolio([decision],panel,1),[])
        rows=agent.replay.pending_batch(8)
        self.assertTrue(all(e.bootstrap_discount==0 and e.reward<0 for e in rows))
        self.assertAlmostEqual(next(e.portfolio_reward for e in rows if e.portfolio_transition),-.02)

    def test_fifteen_second_observations_keep_same_credit_duration(self):
        from test_online_pipeline import PipelineTests
        panel=Panel();panel.dates=(panel.dates[0]+np.arange(8)*np.timedelta64(15,"s"))
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent._init_model_lifecycle(False)
        agent.replay=GlobalReplayBuffer();agent.metrics={"paper_experiences_seen":0}
        agent.window=4;agent.champion=FixedPolicy();agent.paper_account=PaperAccount.in_memory(0,0)
        agent.horizon_kind="bars";agent.horizon_amount=1
        exp=PipelineTests.experience()
        pending=[{**vars(exp),"symbol":"TEST.KS","action":1,"timestamp":str(panel.dates[0]),
            "entry_price":100,"equity_before":2,"symbol_pnl_before":0,
            "credit_observations_target":1,"credit_seconds":60}]
        for i in (1,2,3):
            pending=agent._mature_portfolio(pending,panel,i)
            self.assertEqual(len(pending),1)
            self.assertEqual(len(agent.replay),0)
        self.assertEqual(agent._mature_portfolio(pending,panel,4),[])
        self.assertTrue(all(e.credit_observations==4 for e in agent.replay.pending_batch(8)))

    def test_long_credit_pending_and_successor_survive_restart_and_dual_ack(self):
        panel=Panel();panel.closes[:,0]=[100,99,98,106,107,108,109,110]
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"replay.sqlite3"
            agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
            agent._init_model_lifecycle(False);agent.window=4;agent.champion=FixedPolicy()
            agent.replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            agent.metrics={"paper_experiences_seen":0};agent.horizon_kind="bars";agent.horizon_amount=1
            account=PaperAccount.in_memory(0,0);agent.paper_account=account
            book=account.state["books"]["KRW"]
            book["cash"]-=10_000;book["positions"]["TEST.KS"]={"quantity":100,"average_cost":100}
            book["marks"]["TEST.KS"]=100
            args=agent._window(panel,0);pstate,astate=account.model_inputs(panel,0)
            decision={"features":args[0][0].numpy(),"symbol_ids":panel.symbol_ids,
                "market_ids":panel.market_ids,"asset_ids":panel.asset_ids,"valid_mask":args[4][0].numpy(),
                "portfolio_state":pstate,"account_state":astate,"market_context":args[5][0].numpy(),
                "symbol":"TEST.KS","symbol_index":0,"action":1,"timestamp":str(panel.dates[0]),
                "reward_version":REWARD_VERSION,"equity_before":account.normalized_equity(),
                "symbol_pnl_before":account.symbol_net_pnl("TEST.KS"),"credit_observations_target":3,
                "entry_price":100,"behavior_log_prob":float(np.log(1/3))}
            pending=[decision]
            for i in (1,1,2):
                account.process_bar(panel,i,True)
                pending=agent._mature_portfolio(pending,panel,i)
                self.assertEqual(len(pending),1)
            self.assertEqual(pending[0]["credit_observations_elapsed"],2)
            # Credit duration does not become shorter with finer observations.
            pending[0]["credit_seconds"]=180
            agent.replay.save_pending([],pending)
            self.assertEqual(agent.replay.load_pending("portfolio")[0]["credit_seconds"],180)
            agent.replay.save_pending([],pending)
            agent.replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            pending=agent.replay.load_pending("portfolio")
            account.process_bar(panel,3,True)
            self.assertEqual(agent._mature_portfolio(pending,panel,3),[])
            agent.replay=GlobalReplayBuffer(journal_path=path,dual_learning=True)
            rows=agent.replay.pending_batch(8)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(e.reward>0 and e.credit_observations==3 for e in rows))
            self.assertTrue(all(e.bootstrap_discount==1 and e._bootstrap_inputs for e in rows))
            self.assertEqual(len({e.bootstrap_window_key for e in rows}),1)
            ids=agent.replay.row_ids_for(rows)
            agent.replay.acknowledge_training(dict.fromkeys(ids,1),learner="champion")
            self.assertEqual(len(agent.replay),2)
            agent.replay.acknowledge_training(dict.fromkeys(ids,1),learner="candidate")
            with closing(agent.replay._connect()) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM windows").fetchone()[0],0)

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
        agent._init_model_lifecycle(False)
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
            agent._init_model_lifecycle(False)
            agent.replay=GlobalReplayBuffer(journal_path=root/"replay.sqlite3",dual_learning=True)
            for i in (1,2):agent.replay.enqueue_market_observation(MarketObservation(panel,i,8),True,(.9,.9))
            agent.window=8;agent.stop=MagicMock();agent.stop.is_set.side_effect=[False,False,False,False,True]
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
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent._init_model_lifecycle(False);agent.cfg=cfg
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
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent._init_model_lifecycle(False);agent.cfg=cfg
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

    def test_matured_batch_stores_shared_inputs_once_and_acknowledges_atomically(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            replay=GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3",dual_learning=True)
            panel=Panel()
            base=dict(features=panel.features[:2],symbol_ids=panel.symbol_ids,
                market_ids=panel.market_ids,asset_ids=panel.asset_ids,valid_mask=panel.observed[:2],
                symbol_index=0,action=2,reward=.1,timestamp=str(panel.dates[1]),
                source="paper_account_symbol",reward_version=REWARD_VERSION)
            successor=Experience(**{**base,"timestamp":str(panel.dates[2]),"features":panel.features[:3],
                                    "valid_mask":panel.observed[:3]})
            rows=[];acks=[]
            with closing(replay._connect()) as db,db:
                for i in range(64):
                    row=Experience(**{**base,"timestamp":str(i)})
                    row._bootstrap_experience=successor
                    rows.append(row);acks.append(("portfolio",str(i)))
                    db.execute("INSERT INTO pending_records VALUES(?,?,?)",("portfolio",str(i),b"test"))
            with patch.object(replay,"_store_window",wraps=replay._store_window) as store:
                replay.add_many(rows,pending_acks=acks)
                self.assertEqual(store.call_count,2)
            with closing(replay._connect()) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0],64)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM pending_records").fetchone()[0],0)
                self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0],"ok")
            with closing(replay._connect()) as db,db:
                db.execute("INSERT INTO pending_records VALUES('portfolio','retry',?)",(b"test",))
            with patch.object(replay,"_metadata",side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    replay.add_many([Experience(**base)],pending_acks=[("portfolio","retry")])
            with closing(replay._connect()) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0],64)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM pending_records").fetchone()[0],1)

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
