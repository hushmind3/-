"""Elapsed reward settlement never fabricates a new quote or drops experience."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from stockrl.rewards import closed_market_credit_ready, saved_closing_panel
from stockrl.rewards import _RewardMixin
from stockrl.replay_store import GlobalReplayBuffer
from stockrl.experience import REWARD_VERSION
from stockrl.paper_account import PaperAccount
from test_paper_infrastructure import Panel


class PendingRewardChecks(unittest.TestCase):
    def test_closed_reward_metadata_survives_replay_reload(self):
        panel,agent,decision=self.setup_case()
        with tempfile.TemporaryDirectory() as directory:
            journal=Path(directory)/"replay.sqlite3"
            agent.replay=GlobalReplayBuffer(journal_path=journal)
            agent._mature_portfolio([decision],panel,2)
            restored=GlobalReplayBuffer(journal_path=journal)
            self.assertEqual(len(restored.items),2)
            for experience in restored.items:
                self.assertEqual(experience.reward_settlement,"closed_market_last_real_mark")
                self.assertEqual(experience.reward_end_timestamp,str(panel.dates[2]))
                self.assertEqual(experience.reward_quote_timestamp,str(panel.dates[1]))

    def setup_case(self, current="2026-10-01T11:06:00"):
        panel=Panel()
        panel.dates=np.array(["2026-10-01T10:00:00","2026-10-01T10:59:00",current],dtype="datetime64[ns]")
        panel.observed=panel.observed[:3].copy();panel.observed[2,0]=False
        panel.features=panel.features[:3];panel.closes=panel.closes[:3].copy();panel.closes[2,0]=panel.closes[1,0]
        agent=_RewardMixin()
        agent.metrics={};agent.replay=GlobalReplayBuffer()
        agent.paper_account=PaperAccount.in_memory(.001,.0001)
        agent.paper_account.state["books"]["KRW"]["positions"]["TEST.KS"]={"quantity":10,"average_cost":100.0}
        agent.paper_account.state["books"]["KRW"]["marks"]["TEST.KS"]=110.0
        agent._window=lambda p,i:(torch.as_tensor(p.features[:2][None]),torch.as_tensor(p.symbol_ids[None]),torch.as_tensor(p.market_ids[None]),torch.as_tensor(p.asset_ids[None]),torch.as_tensor(p.observed[:2][None]),torch.as_tensor(p.market_context[:2][None]))
        decision={"timestamp":str(panel.dates[0]),"symbol":"TEST.KS","symbol_index":0,"action":1,
            "reward_version":REWARD_VERSION,"credit_observations_target":60,"credit_seconds":3600,
            "features":panel.features[:2],"symbol_ids":panel.symbol_ids,"market_ids":panel.market_ids,"asset_ids":panel.asset_ids,
            "valid_mask":panel.observed[:2],"input_symbols":panel.symbols,"equity_before":2.0,"symbol_pnl_before":0.0}
        return panel,agent,decision

    def test_closed_equity_settles_once_and_keeps_bootstrap(self):
        panel,agent,decision=self.setup_case()
        self.assertTrue(closed_market_credit_ready(decision,panel,2))
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[])
        rows=list(agent.replay.items)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(e.reward>0 for e in rows))
        self.assertTrue(all(e.reward_settlement=="closed_market_last_real_mark" for e in rows))
        self.assertTrue(all(e.bootstrap_discount==1.0 for e in rows))
        self.assertEqual(agent.metrics["champion_closed_market_rewards_settled"],1)
        self.assertEqual(agent.metrics["champion_pending_reward_status"]["total"],0)

    def test_missing_quote_during_open_venue_remains_pending(self):
        panel,agent,decision=self.setup_case("2026-10-01T10:59:30")
        decision["timestamp"]="2026-10-01T09:00:00"
        self.assertFalse(closed_market_credit_ready(decision,panel,2))
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[decision])
        self.assertEqual(len(agent.replay),0)
        self.assertEqual(agent.metrics["champion_pending_reward_status"]["reasons"],{"next_quote":1})

    def test_unfilled_order_waits_for_real_fill(self):
        panel,_,decision=self.setup_case();decision.update(fill_expected=True,fill_seen=False)
        self.assertFalse(closed_market_credit_ready(decision,panel,2))

    def test_unelapsed_horizon_and_unknown_price_are_not_settled(self):
        panel,_,decision=self.setup_case();decision["timestamp"]="2026-10-01T10:59:00"
        self.assertFalse(closed_market_credit_ready(decision,panel,2))
        decision["timestamp"]="2026-10-01T10:00:00";panel.closes[2,0]=np.nan
        self.assertFalse(closed_market_credit_ready(decision,panel,2))

    def test_us_extended_hours_are_kept(self):
        panel,_,decision=self.setup_case("2026-10-02T00:00:00")
        panel.groups["TEST.KS"]=("US","equity")
        self.assertFalse(closed_market_credit_ready(decision,panel,2))  # 20:00 EDT + final-bar buffer
        panel.dates[2]=np.datetime64("2026-10-02T00:06:00")
        self.assertTrue(closed_market_credit_ready(decision,panel,2))

    def test_removed_closed_symbol_reuses_durable_input(self):
        panel,agent,decision=self.setup_case()
        decision['entry_price']=110.0
        panel.symbols=['OTHER']; panel.groups={'OTHER':('US','equity')}
        recovered=saved_closing_panel([decision],panel,2)
        self.assertIs(recovered.saved_reward_input.features,decision['features'])
        self.assertEqual(str(recovered.dates[0]),decision['timestamp'])
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[])
        self.assertEqual(len(agent.replay.items),2)
        self.assertTrue(all(e._bootstrap_experience.features is decision['features'] for e in agent.replay.items))

    def test_missing_open_symbol_is_not_credited(self):
        panel,agent,decision=self.setup_case('2026-10-01T10:59:30')
        decision['timestamp']='2026-10-01T09:00:00';decision['entry_price']=110.0
        panel.symbols=['OTHER'];panel.groups={'OTHER':('US','equity')}
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[decision])
        self.assertEqual(len(agent.replay.items),0)

    def test_unfilled_closing_order_expires_without_fake_fill(self):
        panel,agent,decision=self.setup_case()
        decision.update(entry_price=110.0,fill_expected=True,fill_seen=False)
        panel.symbols=['OTHER'];panel.groups={'OTHER':('US','equity')}
        agent.paper_account.state['pending']['TEST.KS']={'date':decision['timestamp']}
        self.assertIsNotNone(saved_closing_panel([decision],panel,2))
        self.assertEqual(agent._mature_portfolio([decision],panel,2),[])
        self.assertNotIn('TEST.KS',agent.paper_account.state['pending'])
        self.assertTrue(all(not row.trade_executed and row.reward==0 for row in agent.replay.items))
        self.assertEqual(agent.metrics['champion_expired_session_orders'],1)

if __name__=="__main__": unittest.main()
