"""Shared account, reward, replay, market-data and GPU scheduling regressions."""
from pathlib import Path
from contextlib import closing
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json, queue, threading, unittest, sqlite3
import numpy as np
import torch
from stockrl.experience import Experience, IncrementalMarketCSV, MarketObservation, REWARD_VERSION
from stockrl.replay_store import GlobalReplayBuffer
from stockrl.market_panel import GlobalMarketPanel, MARKET_CONTEXT_FEATURES as CONTEXT_FEATURES
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

class SharedPaperTests(unittest.TestCase):
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

    def test_discarded_invalid_rows_do_not_remain_in_daily_backlog(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            replay=GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3",dual_learning=True)
            replay.add(self.experience())
            ids=replay.row_ids_for(replay.pending_batch(8))
            replay.quarantine(ids,"invalid historical input")
            replay.discard_row_ids(ids)
            day=replay.stats()["daily"][0]
            self.assertEqual((day["enqueued"],day["completed"]),(1,0))
            self.assertEqual((day["remaining"],day["remaining_total"],day["blocked"]),(0,0,0))
            self.assertEqual(day["removed_without_completion"],1)

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

    def test_reader_preserves_closed_equity_window(self):
        import pandas as pd
        dates=pd.date_range("2026-09-30",periods=400,freq="min")
        active=pd.DataFrame({"date":dates,"symbol":"ACTIVE","asset_class":"equity"})
        closed=pd.DataFrame({"date":dates[:150],"symbol":"CLOSED.KS","asset_class":"equity"})
        reader=IncrementalMarketCSV("unused.csv",retain_timestamps=128)
        reader.processed_through=str(dates[350])
        trimmed=reader._trim(pd.concat([active,closed],ignore_index=True))
        kept=trimmed.loc[trimmed.symbol=="CLOSED.KS"]
        self.assertEqual(len(kept),128)
        self.assertEqual(list(kept.date),list(dates[22:150]))

    def test_feed_compaction_carries_closed_equity_quotes(self):
        import pandas as pd
        from stockrl.live_feed import AppendOnlyMarketCSV
        with TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"market.csv"; (path.parent/"agent").mkdir()
            dates=pd.date_range("2026-09-30",periods=800,freq="min",tz="UTC")
            atomic_json({"last_timestamp":dates[700].isoformat()},path.parent/"agent"/"live_cursor.json")
            writer=AppendOnlyMarketCSV(path);writer.COMPACT_CSV_BYTES=1;writer._next_csv_compaction=1
            try:
                rows=[{"date":stamp.isoformat(),"symbol":"ACTIVE","market":"US","asset_class":"equity","close":100} for stamp in dates]
                rows += [{"date":stamp.isoformat(),"symbol":"CLOSED.KS","market":"KRX","asset_class":"equity","close":110} for stamp in dates[:200]]
                writer.append(rows)
            finally:
                writer.close()
            kept=pd.read_csv(path).query("symbol=='CLOSED.KS'")
            self.assertEqual(len(kept),128)
            self.assertEqual(list(pd.to_datetime(kept.date,utc=True)),list(dates[72:200]))
            self.assertTrue((kept.close==110).all())

    def test_pending_ack_accepts_both_call_styles(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            replay=GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3")
            first={"timestamp":"T1","symbol":"A"}; second={"timestamp":"T2","symbol":"A"}
            replay.save_pending([first,second],[])
            replay.acknowledge_pending("regular",first)
            replay.acknowledge_pending("regular","T2|A")
            self.assertEqual(replay.load_pending("regular"),[])

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

    def test_compaction_reuses_free_pages_and_shrinks_when_queue_drains(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"replay.sqlite3"
            replay = GlobalReplayBuffer(journal_path=path)
            replay.add(self.experience())
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE scratch(data BLOB)")
                db.execute("INSERT INTO scratch VALUES(zeroblob(8388608))")
                db.execute("DELETE FROM scratch")
            traced = []
            connect = replay._connect
            def trace_connect():
                db = connect(); db.set_trace_callback(traced.append); return db
            with patch.object(replay, "_connect", side_effect=trace_connect):
                replay.compact()
                self.assertFalse(any(sql == "VACUUM" for sql in traced))
                self.assertEqual(len(replay), 1)
                uses = dict.fromkeys(replay.row_ids_for(replay.pending_batch(8)), 1)
                replay.acknowledge_training(uses)
                self.assertTrue(any(sql == "VACUUM" for sql in traced))
                self.assertEqual(len(replay), 0)
