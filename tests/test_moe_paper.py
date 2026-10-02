import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from stockrl.moe_paper import TradingMoEPaper


def panel():
    return SimpleNamespace(dates=np.array(["2025-10-29T14:00","2025-10-29T14:01","2025-10-29T15:02"],dtype="datetime64[ns]"),
        symbols=["AAPL"],groups={"AAPL":("US","equity")},observed=np.ones((3,1),bool),
        closes=np.array([[100.],[101.],[102.]]),features=np.zeros((3,1,17),np.float32),
        symbol_ids=np.array([0]),market_ids=np.array([0]),asset_ids=np.array([0]))


class PaperBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.bridge=TradingMoEPaper(Path(self.temp.name));self.panel=panel()
        self.result={"as_of":str(self.panel.dates[0]),"trading_output":{"executable":False,
            "actions":{"AAPL":"BUY"},"target_weights":{"AAPL":.4},"cash_weights_by_currency":{"USD":.6}}}

    def tearDown(self):self.temp.cleanup()

    def test_paper_guard_and_live_guard(self):
        self.bridge.advance(self.panel,0)
        self.assertFalse(self.bridge.submit(self.result,self.panel,0)["paper_executable"])
        self.assertFalse(self.bridge.paper_account.state["pending"])
        self.bridge.submit(self.result,self.panel,0,paper_executable=True)
        self.assertFalse(self.result["trading_output"]["executable"])
        self.assertFalse(self.bridge.advance(self.panel,0))
        self.assertEqual(len(self.bridge.advance(self.panel,1)),1)
        self.bridge.advance(self.panel,2)
        self.assertLess(self.bridge.paper_account.state["books"]["USD"]["cash"],10000)
        self.assertEqual(self.bridge.replay.stats()["total"],2)
        self.assertEqual(self.bridge.replay.stats()["unsupported"],0)

    def test_pending_survives_restart(self):
        self.bridge.advance(self.panel,0);self.bridge.submit(self.result,self.panel,0,paper_executable=True)
        loaded=TradingMoEPaper(Path(self.temp.name))
        self.assertEqual(len(loaded.pending),1)
        self.assertEqual(len(loaded.advance(self.panel,1)),1)
        loaded.advance(self.panel,2)
        self.assertEqual(loaded.replay.stats()["total"],2)

    def test_stale_asof_rejected(self):
        with self.assertRaises(ValueError):self.bridge.submit(self.result,self.panel,1,paper_executable=True)

    def test_invalid_allocation_rejected(self):
        self.result["trading_output"]["target_weights"]["AAPL"]=-1
        with self.assertRaises(ValueError):self.bridge.submit(self.result,self.panel,0,paper_executable=True)


if __name__=="__main__":unittest.main()
