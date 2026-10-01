import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from stockrl.replay_store import GlobalReplayBuffer
from stockrl.online.optimizer_state import remember_optimizer, restore_optimizer
from stockrl.online.replay_updates import install_replay_updates
import test_online_pipeline as pipeline_checks


class ThroughputMaintenanceTests(unittest.TestCase):
    def test_optimizer_moments_continue_without_retaining_gpu_parameters(self):
        owner = SimpleNamespace()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        first = torch.optim.AdamW([parameter])
        parameter.sum().backward(); first.step()
        remember_optimizer(owner, "candidate", first, 2)
        other = torch.nn.Parameter(parameter.detach().clone())
        resumed = torch.optim.AdamW([other])
        self.assertTrue(restore_optimizer(owner, "candidate", resumed, 2))
        other.sum().backward(); resumed.step()
        self.assertEqual(resumed.state[other]["step"].item(), 2)
        self.assertFalse(restore_optimizer(owner, "candidate", resumed, 3))
        self.assertTrue(all(not value.is_cuda for state in owner._optimizer_states["candidate"]["state"]["state"].values()
                            for value in state.values() if isinstance(value, torch.Tensor)))

    def test_bulk_ack_retains_untrained_rows_and_skips_unnecessary_cleanup(self):
        with TemporaryDirectory() as directory:
            replay = GlobalReplayBuffer(journal_path=Path(directory)/"replay.sqlite3", dual_learning=True)
            replay.add_many([pipeline_checks.PipelineTests.experience("2026-09-30T01:00:00"),
                             pipeline_checks.PipelineTests.experience("2026-09-30T01:01:00")])
            install_replay_updates(replay)
            uses = dict.fromkeys(replay.row_ids_for(replay.pending_batch(8)), 1)
            with patch.object(replay, "_collect_unused_windows", wraps=replay._collect_unused_windows) as collect:
                self.assertEqual(replay.acknowledge_training(uses, learner="champion"), 0)
                self.assertEqual(collect.call_count, 0)
                self.assertEqual(len(replay), 2)
                self.assertEqual(replay.acknowledge_training(uses, learner="candidate"), 2)
                self.assertEqual(collect.call_count, 1)
                self.assertEqual(replay.acknowledge_training(uses, learner="candidate"), 0)
                self.assertEqual(len(replay), 0)

    def test_compaction_reuses_free_pages_and_shrinks_when_queue_drains(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"replay.sqlite3"
            replay = GlobalReplayBuffer(journal_path=path)
            replay.add(pipeline_checks.PipelineTests.experience())
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


if __name__ == "__main__":
    unittest.main()
