"""Regressions for Python cleanup, without loading any expert checkpoint."""
import gc
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from stockrl import expert_backends, moe_native, state_io
from stockrl.moe_paper import TradingMoEPaper


RUNNER = """def run_native(expert, root, data, device='cpu'):
    models = [subagent()]
    loaded_seconds = 0.0
    return models[0](data)
"""


class Shift(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, value):
        return value + self.offset


class NativePreparationTests(unittest.TestCase):
    def setUp(self):
        moe_native._compiled_runner.cache_clear()

    def tearDown(self):
        moe_native._compiled_runner.cache_clear()

    def test_compiled_code_reused_without_reusing_model_or_input(self):
        with patch.object(moe_native.ast, 'parse', wraps=moe_native.ast.parse) as parse:
            first = moe_native.native_call('macro', '.', 2, modules=[Shift(3)], runner_source=RUNNER)
            second = moe_native.native_call('macro', '.', 4, modules=[Shift(7)], runner_source=RUNNER)
            self.assertEqual((first, second), (5, 11))
            self.assertEqual(parse.call_count, 1)
        self.assertEqual(moe_native._compiled_runner.cache_info().hits, 1)

    def test_load_only_and_runner_revisions_keep_separate_bindings(self):
        model = Shift(3)
        with patch.object(torch, 'load', side_effect=AssertionError('must not reopen weights')):
            self.assertIs(moe_native.native_call('macro', '.', 2, modules=[model],
                load_only=True, runner_source=RUNNER)[0], model)
            changed = RUNNER.replace('return models[0](data)', 'return 2 * models[0](data)')
            self.assertEqual(moe_native.native_call('macro', '.', 2, modules=[model], runner_source=changed), 10)
        self.assertEqual(moe_native._compiled_runner.cache_info().misses, 2)

    def test_compiled_cache_has_a_fixed_bound(self):
        for index in range(12):
            moe_native._compiled_runner(RUNNER + '# revision %s\n' % index)
        self.assertEqual(moe_native._compiled_runner.cache_info().currsize, 8)

    def test_source_module_reuses_classes_and_invalidates_changed_source(self):
        name = '_stockrl_cleanup_fixture'
        previous = sys.modules.pop(name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'native.py'
                path.write_text('class Native: value = 12\n', encoding='utf-8')
                first = expert_backends.source_module(name, path)
                self.assertIs(expert_backends.source_module(name, path), first)
                path.write_text('class Native: value = 1234\n', encoding='utf-8')
                changed = expert_backends.source_module(name, path)
                self.assertIsNot(changed, first)
                self.assertEqual(changed.Native.value, 1234)
        finally:
            sys.modules.pop(name, None)
            if previous is not None: sys.modules[name] = previous

    def test_toto_uses_one_loader_without_optional_bridge_imports(self):
        names = ['research_toto', 'research_toto.configuration', 'research_toto.model']
        previous = {name: sys.modules.pop(name, None) for name in names}
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'configuration.py').write_text('VALUE = 7\n', encoding='utf-8')
                (root / 'model.py').write_text(
                    'from gluonts.torch import missing\n'
                    'from .configuration import VALUE\n'
                    'class Toto2GluonTSModel: pass\n'
                    'class _FnImputation: pass\n'
                    'class Toto2Model: value = VALUE\n', encoding='utf-8')
                first = expert_backends.toto_native_module(root)
                self.assertIs(expert_backends.toto_native_module(root), first)
                self.assertEqual(first.Toto2Model.value, 7)
                self.assertFalse(hasattr(first, '_FnImputation'))
                (root / 'configuration.py').write_text('VALUE = 999\n', encoding='utf-8')
                self.assertEqual(expert_backends.toto_native_module(root).Toto2Model.value, 999)
        finally:
            for name in names:
                sys.modules.pop(name, None)
                if previous[name] is not None: sys.modules[name] = previous[name]


class StatePersistenceTests(unittest.TestCase):
    def test_json_reader_preserves_values_and_does_not_share_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            fallback = {'queue': []}
            missing = state_io.read_json(path, fallback)
            missing['queue'].append(1)
            self.assertEqual(fallback, {'queue': []})
            path.write_text('{broken', encoding='utf-8')
            self.assertEqual(state_io.read_json(path), {})
            state_io.atomic_json({'value': [1, 2]}, path)
            self.assertEqual(state_io.read_json(path), {'value': [1, 2]})

    def test_concurrent_atomic_writers_release_per_path_locks(self):
        gc.collect()
        baseline = len(state_io._locks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda index: state_io.atomic_json({'index': index}, path), range(32)))
            self.assertIn(state_io.read_json(path)['index'], range(32))
            self.assertFalse(path.with_name(path.name + '.tmp').exists())
            for index in range(25): state_io.atomic_json({}, Path(directory) / ('%s.json' % index))
        gc.collect()
        self.assertEqual(len(state_io._locks), baseline)


class SharedPaperInputTests(unittest.TestCase):
    def test_symbols_share_owned_input_block_and_ledger_replay_still_mature(self):
        panel = SimpleNamespace(
            dates=np.array(['2025-10-29T14:00', '2025-10-29T14:01', '2025-10-29T15:02'], dtype='datetime64[ns]'),
            symbols=['AAPL', 'MSFT'], groups={'AAPL': ('US', 'equity'), 'MSFT': ('US', 'equity')},
            observed=np.ones((3, 2), bool), closes=np.array([[100., 200.], [101., 201.], [102., 202.]]),
            features=np.zeros((3, 2, 17), np.float32), symbol_ids=np.array([0, 1]),
            market_ids=np.array([0, 0]), asset_ids=np.array([0, 0]))
        result = {'as_of': str(panel.dates[0]), 'currencies': {'AAPL': 'USD', 'MSFT': 'USD'},
            'trading_output': {'actions': {'AAPL': 'BUY', 'MSFT': 'BUY'},
                'target_weights': {'AAPL': .3, 'MSFT': .3}, 'cash_weights_by_currency': {'USD': .4}}}
        with tempfile.TemporaryDirectory() as directory:
            bridge = TradingMoEPaper(directory)
            bridge.advance(panel, 0)
            self.assertTrue(bridge.submit(result, panel, 0, paper_executable=True)['orders'])
            self.assertEqual(len(bridge.pending), 2)
            first, second = bridge.pending
            for key in ('features', 'valid_mask', 'symbol_ids', 'market_ids', 'asset_ids'):
                self.assertIs(first[key], second[key])
            panel.features[0] = 99
            panel.symbol_ids[0] = 77
            self.assertEqual(first['features'].max(), 0)
            self.assertEqual(first['symbol_ids'][0], 0)
            panel.features[0] = 0
            panel.symbol_ids[0] = 0
            resumed = TradingMoEPaper(directory)
            self.assertEqual(len(resumed.pending), 2)
            fills = resumed.advance(panel, 1)
            self.assertEqual(len(fills), 2)
            resumed.advance(panel, 2)
            self.assertGreater(resumed.replay.stats()['total'], 0)
            self.assertGreater(resumed.paper_account.snapshot()['books']['USD']['holdings_value'], 0)


if __name__ == '__main__':
    unittest.main()
