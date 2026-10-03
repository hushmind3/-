"""Shared market/reward/experience contracts and cold command imports."""
import json
import os
import pickle
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SharedInfrastructureTests(unittest.TestCase):
    def child(self, source):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUTF8='1')
        result = subprocess.run([sys.executable, '-c', source], cwd=ROOT, env=environment,
            capture_output=True, text=True, encoding='utf-8', timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_compatibility_paths_reference_the_same_shared_objects(self):
        from stockrl import experience, market_panel, rewards
        from stockrl.online import data, rewards as old_rewards
        self.assertIs(data.Experience, experience.Experience)
        self.assertIs(data.MarketObservation, experience.MarketObservation)
        self.assertIs(data.parse_horizon, experience.parse_horizon)
        self.assertIs(old_rewards._RewardMixin, rewards._RewardMixin)
        self.assertIs(old_rewards.saved_closing_panel, rewards.saved_closing_panel)

    def test_stored_old_experience_class_reference_still_resolves(self):
        from stockrl.experience import Experience
        self.assertIs(pickle.loads(b'cstockrl.online.data\nExperience\n.'), Experience)
        self.assertIn('reward_settlement', {field.name for field in fields(Experience)})
        self.assertEqual(Experience.__module__, 'stockrl.experience')

    def test_horizon_contract_unchanged(self):
        from stockrl.experience import parse_horizon
        self.assertEqual(parse_horizon('1m'), ('seconds', 60))
        self.assertEqual(parse_horizon('2h'), ('seconds', 7200))
        with self.assertRaises(ValueError): parse_horizon('invalid')

    def test_shared_paper_infrastructure_has_no_legacy_or_torch_import(self):
        report = self.child("""
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'torch', 'stockrl.global_transformer', 'stockrl.global_online', 'stockrl.market_training'}:
            raise RuntimeError('unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
from stockrl.market_panel import GlobalMarketPanel
from stockrl.experience import Experience, MarketObservation
from stockrl.moe_paper import TradingMoEPaper
import pandas as pd
frame = pd.DataFrame([dict(date='2025-10-29T14:0'+str(i), symbol='AAPL', market='US',
    asset_class='equity', open=100+i, high=101+i, low=99+i, close=100+i, volume=10) for i in range(3)])
panel = GlobalMarketPanel('unused.csv', raw_frame=frame, symbol_map={'US|equity|AAPL': 0})
assert panel.market_context.shape == (3, 16)
observation = MarketObservation(panel, 2, 3)
print(json.dumps({'symbols': panel.symbols, 'rows': len(observation.dates), 'torch': 'torch' in sys.modules}))
""")
        self.assertEqual(report, {'symbols': ['AAPL'], 'rows': 3, 'torch': False})

    def test_cli_feed_help_and_disabled_train_do_not_load_models(self):
        report = self.child("""
import json, sys
from stockrl import cli
sys.argv = ['stockrl', 'live-feed', '--help']
try: cli.main()
except SystemExit as error: assert error.code == 0
sys.argv = ['stockrl', 'train']
try: cli.main()
except SystemExit as error: assert error.code == 2
print(json.dumps({'torch': 'torch' in sys.modules, 'legacy': 'stockrl.global_online' in sys.modules,
    'numpy': 'numpy' in sys.modules}))
""")
        self.assertEqual(report, {'torch': False, 'legacy': False, 'numpy': False})

    def test_actual_feed_imports_and_moe_module_are_independent_of_old_models(self):
        report = self.child("""
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'stockrl.global_transformer', 'stockrl.global_online', 'stockrl.market_training'}:
            raise RuntimeError('unexpected legacy import: ' + fullname)
sys.meta_path.insert(0, Block())
import stockrl.cli, stockrl.live_feed, stockrl.mock_feed
feed_loaded_torch = 'torch' in sys.modules
import stockrl.trading_moe
print(json.dumps({'feed_loaded_torch': feed_loaded_torch, 'moe': hasattr(stockrl.trading_moe, 'TradingMoE'),
    'legacy': 'stockrl.global_transformer' in sys.modules}))
""")
        self.assertEqual(report, {'feed_loaded_torch': False, 'moe': True, 'legacy': False})


if __name__ == '__main__': unittest.main()
