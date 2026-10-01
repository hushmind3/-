"""The faster shared loss must preserve the scalar objective and gradients."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock
from stockrl.global_online import OnlineGlobalAgent
from stockrl.online.losses import shared_experience_losses


class SharedLossChecks(unittest.TestCase):
    def test_direct_sdpa_matches_mha_outputs_and_gradients(self):
        from stockrl.global_transformer import TransformerBlock,TransformerConfig
        torch.manual_seed(81)
        block=TransformerBlock(TransformerConfig(d_model=32,n_heads=4,n_layers=6))
        x=torch.randn(3,12,32,requires_grad=True)
        mask=torch.zeros(3,12,dtype=torch.bool);mask[:,-2:]=True
        params=(x,)+tuple(block.parameters())
        block._stockrl_sdpa_enabled=False
        old=block(x,mask); grads=torch.autograd.grad(old.square().mean(),params)
        block._stockrl_sdpa_enabled=True
        new=block(x,mask); newer=torch.autograd.grad(new.square().mean(),params)
        torch.testing.assert_close(new,old,rtol=2e-5,atol=2e-6)
        for actual,expected in zip(newer,grads):
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_frozen_prefix_reuse_preserves_outputs_gradients_and_invalidates(self):
        from stockrl.global_transformer import GlobalMarketTransformer,TransformerConfig
        from stockrl.online.prefix_cache import FrozenPrefixCache,prefix_input
        torch.manual_seed(11)
        cfg=TransformerConfig(d_model=16,n_heads=4,n_layers=6,ff_mult=2,max_symbols=8,max_seq_len=8)
        model=GlobalMarketTransformer(cfg).train()
        for p in model.parameters():p.requires_grad_(False)
        for block in model.blocks[-4:]:
            for p in block.parameters():p.requires_grad_(True)
        features=np.random.default_rng(11).normal(size=(4,3,17)).astype(np.float32)
        row=SimpleNamespace(features=features,symbol_ids=np.arange(3),market_ids=np.zeros(3),
            asset_ids=np.zeros(3),valid_mask=np.array([[1,1,0],[1,1,0],[1,1,0],[1,1,0]],dtype=bool))
        args=[torch.as_tensor(row.features)[None],torch.as_tensor(row.symbol_ids)[None],
            torch.as_tensor(row.market_ids,dtype=torch.long)[None],torch.as_tensor(row.asset_ids,dtype=torch.long)[None],torch.as_tensor(row.valid_mask)[None]]
        params=[p for p in model.parameters() if p.requires_grad]
        ordinary=model(*args);reference=torch.autograd.grad(sum(x.sum() for x in ordinary),params)
        cache=FrozenPrefixCache(max_bytes=100000,max_entries=2)
        for _ in range(2):
            with prefix_input(model,row,cache):actual=model(*args)
            for a,b in zip(actual,ordinary):torch.testing.assert_close(a,b)
            for a,b in zip(torch.autograd.grad(sum(x.sum() for x in actual),params),reference):torch.testing.assert_close(a,b)
        self.assertEqual((cache.hits,cache.misses),(1,1))
        with torch.no_grad():model.input_proj.weight.add_(.001)
        with prefix_input(model,row,cache):model(*args)
        self.assertEqual(cache.misses,2)
        self.assertLessEqual(cache.bytes,cache.max_bytes)

    def test_runtime_controller_refresh_preserves_state(self):
        from stockrl.online.runtime_updates import RuntimeUpdates
        from stockrl.operating_rules import operating_rules
        with TemporaryDirectory() as directory:
            path=Path(directory)/"rules.json";rules=operating_rules()
            path.write_text(json.dumps(rules),encoding="utf-8")
            agent=OnlineGlobalAgent.__new__(OnlineGlobalAgent)
            agent.metrics={};agent.operating_rules=rules
            original=agent.runtime_updates=RuntimeUpdates(agent,path,watch_code=False)
            agent._refresh_runtime_update_driver()
            self.assertIs(agent.runtime_updates,original)
            self.assertEqual(agent.runtime_updates.path,path)

    def test_runtime_settings_apply_without_replacing_models(self):
        from stockrl.online.runtime_updates import RuntimeUpdates
        from stockrl.operating_rules import operating_rules
        with TemporaryDirectory() as directory:
            path=Path(directory)/"rules.json"
            rules=operating_rules();path.write_text(json.dumps(rules),encoding="utf-8")
            model=object()
            agent=SimpleNamespace(metrics={},operating_rules=rules.copy(),validation_active=False,
                champion=model,batch_size=256,updates_per_candidate=8,
                candidate_live_inference_lock=SimpleNamespace(preopen_learning=True))
            updater=RuntimeUpdates(agent,path,watch_code=False)
            rules.update(training_batch_size=128,training_optimizer_steps=4,
                reward_credit_seconds=1800,goal_target_multiple=20)
            path.write_text(json.dumps(rules),encoding="utf-8");updater.stamp=None
            updater.poll(agent,force=True)
            self.assertIs(agent.champion,model)
            self.assertEqual((agent.batch_size,agent.updates_per_candidate,agent.reward_credit_seconds),(128,4,1800))
            self.assertEqual(agent.operating_rules["goal_target_multiple"],10)
            self.assertEqual(agent.metrics["runtime_updates"]["deferred_rules"]["goal_target_multiple"],20)
            path.write_text('{',encoding="utf-8");updater.stamp=None;updater.poll(agent,force=True)
            self.assertEqual(agent.batch_size,128)
            self.assertEqual(agent.metrics["runtime_updates"]["status"],"rejected")

    def test_active_trial_keeps_original_comparison_rules(self):
        from stockrl.online.runtime_updates import RuntimeUpdates
        from stockrl.operating_rules import operating_rules
        with TemporaryDirectory() as directory:
            path=Path(directory)/"rules.json";rules=operating_rules()
            path.write_text(json.dumps(rules),encoding="utf-8")
            agent=SimpleNamespace(metrics={},operating_rules=rules.copy(),validation_active=True,
                validation_window_bars=390,daily_promotion=True)
            updater=RuntimeUpdates(agent,path,watch_code=False)
            rules["validation_min_market_minutes"]=500
            path.write_text(json.dumps(rules),encoding="utf-8");updater.stamp=None
            updater.poll(agent,force=True)
            self.assertEqual(agent.validation_window_bars,390)
            agent.validation_active=False;updater.poll(agent,force=True)
            self.assertEqual(agent.validation_window_bars,500)

    def test_web_handoff_preserves_worker_identity(self):
        from stockrl.web.workers import handoff,adopt
        with TemporaryDirectory() as directory:
            root=Path(directory);profile=root/"live";profile.mkdir()
            process=MagicMock();process.pid=12345;process.create_time.return_value=12.5
            process.cmdline.return_value=["python","-m","stockrl","agent"]
            process.is_running.return_value=True;process.status.return_value="running"
            worker=MagicMock(pid=12345);worker.poll.return_value=None
            original=SimpleNamespace(runtime=root,profile=profile,mode="live",run_requested=True,children={"agent":worker})
            renewed=SimpleNamespace(runtime=root)
            with patch("stockrl.web.workers.psutil.Process",return_value=process):
                handoff(original);self.assertTrue(adopt(renewed))
            self.assertEqual(renewed.children["agent"].pid,12345)
            process.terminate.assert_not_called();process.kill.assert_not_called()
            self.assertFalse((root/"web_workers.json").exists())

    def test_invalid_hot_code_keeps_existing_learner(self):
        from stockrl.online.runtime_updates import RuntimeUpdates
        from stockrl.online.learning import _LearningMixin
        original=_LearningMixin._train_model
        with self.assertRaises(SyntaxError):
            RuntimeUpdates._apply_code(None,{"stockrl.online.learning":(Path("learning.py"),"def broken(:",1)})
        self.assertIs(_LearningMixin._train_model,original)

    def test_values_and_gradients_match_scalar_for_mixed_experiences(self):
        torch.manual_seed(31)
        agent = OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device = torch.device("cpu")
        agent.metrics = {"nonfinite_updates": 0}
        rows = []
        for i in range(48):
            rows.append(SimpleNamespace(
                symbol_index=i % 7, action=i % 3, reward=(i - 24) / 1000,
                goal_reward_points=100.0 if i == 3 else 0.0,
                features=np.zeros((4, 7, 17)), bootstrap_discount=.97 if i % 2 else 0.0,
                bootstrap_symbol_index=(i + 1) % 7,
                portfolio_transition=bool(i % 2), portfolio_value_transition=bool(i % 3),
                portfolio_reward=(i - 12) / 1000, portfolio_goal_reward_points=0.0,
                behavior_log_prob=None if i % 4 else -1.2, trade_executed=bool(i % 5),
                source="teacher" if i % 6 == 0 else "paper"))
        future = [torch.randn(1, 7) for _ in rows]
        logits = torch.randn(1, 7, 3, requires_grad=True)
        values = torch.randn(1, 7, requires_grad=True)
        allocations = torch.rand(1, 8, requires_grad=True)
        scalar = torch.stack([agent._experience_loss(logits, values, allocations, e, s)
                              for e, s in zip(rows, future)])
        grads = torch.autograd.grad(scalar.sum(), (logits, values, allocations), retain_graph=True)
        vector, accepted, rejected = shared_experience_losses(logits, values, allocations, rows, future)
        torch.testing.assert_close(vector, scalar, rtol=2e-6, atol=2e-6)
        for actual, expected in zip(torch.autograd.grad(vector.sum(), (logits, values, allocations)), grads):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        self.assertEqual(len(accepted), len(rows))
        self.assertEqual(rejected, 0)

    def test_nonfinite_row_does_not_discard_other_experiences(self):
        agent = OnlineGlobalAgent.__new__(OnlineGlobalAgent)
        agent.device = torch.device("cpu"); agent.metrics = {"nonfinite_updates": 0}
        def row(i):
            return SimpleNamespace(symbol_index=i, action=1, reward=.01, goal_reward_points=0,
                features=np.zeros((4, 2, 17)), bootstrap_discount=0, portfolio_transition=False,
                portfolio_value_transition=False, portfolio_reward=None, portfolio_goal_reward_points=0,
                behavior_log_prob=None, trade_executed=False, source="paper")
        logits = torch.tensor([[[float("nan"), 0., 0.], [0., 1., 2.]]], requires_grad=True)
        values = torch.zeros(1, 2, requires_grad=True)
        rows = [row(0), row(1)]
        result, accepted, rejected = shared_experience_losses(logits, values, None, rows, [None, None])
        torch.testing.assert_close(result[0], agent._experience_loss(logits, values, None, rows[1]))
        self.assertEqual(accepted, [rows[1]])
        self.assertEqual(rejected, 1)


if __name__ == "__main__":
    unittest.main()
