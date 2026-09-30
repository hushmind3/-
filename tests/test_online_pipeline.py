"""Regression checks for sparse market inputs and the real portfolio ledger."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from stockrl.global_online import GlobalReplayBuffer, MarketObservation, OnlineGlobalAgent, REWARD_VERSION
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
    def test_replay_learning_resumes_without_new_row_counter_and_is_bounded(self):
        agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.replay=GlobalReplayBuffer()
        for row_id in (1,2):
            row=SimpleNamespace(source="paper_account_symbol",reward_version=REWARD_VERSION,
                features=np.zeros((4,2,17)),portfolio_state=None,account_state=None,
                multiscale_state=None)
            agent.replay.items.append(row)
            agent.replay.row_ids[id(row)]=row_id
        agent.min_replay=2; agent.batch_size=2; agent.candidate_replay_passes=2
        agent.candidate_retry_after=0
        for uses,expected in ((1,1),(2,0)):
            agent.metrics={"observation_caught_up":True,"paper_experiences_seen":2,
                           "paper_experiences_since_candidate":0}
            agent.candidate_replay_uses={1:uses,2:uses}
            agent.stop=SimpleNamespace(wait=Mock(side_effect=[False,True]))
            with patch.object(agent,"_train_candidate") as train:
                agent._learner()
                self.assertEqual(train.call_count,expected)
            self.assertEqual(agent.metrics["candidate_eligible_replay_count"],2 if uses==1 else 0)

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
