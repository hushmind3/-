"""Champion/Candidate roles remain operational after the Transformer retirement."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from stockrl.web.runtime import Supervisor
from stockrl.paper_account import PaperAccount
from stockrl.state_io import atomic_json

ROOT=Path(__file__).resolve().parents[1]

class MoELifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runtime')
        self.root=Path(self.temp.name)
        self.supervisor=Supervisor.__new__(Supervisor)
        s=self.supervisor
        s.runtime=self.root;s.profile=self.root/'live';s.profile.mkdir()
        s.mode='live';s.horizon='1m';s.device='auto';s.fee=.001;s.model_dir=self.root/'models'
        s.model_dir.mkdir();s.lock=threading.RLock();s.account_reset_lock=threading.Lock()
        s.model_enabled={'champion':False,'candidate':False}
        s.model_request_versions={};s.model_families=dict.fromkeys(s.model_enabled,'trading_moe')
        s.run_requested=True;s.stopping=False;s.restart_request=None;s.children={}
        s.settings_path=self.root/'settings.json';s.config=self.root/'instruments.json'
        s.autonomy_enabled=True;s.observe_enabled=True;s.learning_enabled=True
        s.log_tail=[];s.log_handles={};s.moe_model_workers={}
        s._market_row_cache={'path':None,'offset':0,'lines':0};s._latest_csv_cache={}
        s._gpu_snapshot={'sampled':0}
        self.workers={}
        for role in s.model_enabled:
            worker=Mock()
            worker.state=s.profile/'agent'/(role+'_moe');worker.state.mkdir(parents=True)
            worker.checkpoint=s.model_dir/'champion.pt';worker.runner_script='run_native_vertical_trading.py'
            worker.status.return_value={'status':'stopped','alive':False,'load_count':0}
            worker.process.return_value=None
            self.workers[role]=worker
        s._moe_model_worker=Mock(side_effect=self.workers.__getitem__)
        s._physical_gpu=Mock(return_value={})

    def tearDown(self):self.temp.cleanup()

    def test_champion_and_candidate_start_and_stop_independently(self):
        s=self.supervisor;s._launch=Mock()
        self.assertTrue(s.set_model('champion',True)['ok'])
        self.assertEqual(s.model_enabled,{'champion':True,'candidate':False})
        self.assertTrue(s.set_model('candidate',True)['ok'])
        self.assertEqual(s.model_enabled,{'champion':True,'candidate':True})
        s.set_model('champion',False)
        self.assertEqual(s.model_enabled,{'champion':False,'candidate':True})
        self.workers['champion'].stop.assert_called_once()
        self.workers['candidate'].stop.assert_not_called()
        self.assertEqual([call.args[0] for call in s._launch.call_args_list],['champion','candidate'])

    def test_running_role_never_loads_twice(self):
        s=self.supervisor;s._launch=Mock()
        self.workers['champion'].status.return_value={'status':'running','alive':True}
        self.assertTrue(s.set_model('champion',True)['already_requested'])
        self.assertTrue(s.set_model('champion',True)['already_requested'])
        s._launch.assert_not_called()

    def test_start_failure_rolls_back_role_flags(self):
        s=self.supervisor;s._launch=Mock(side_effect=RuntimeError('missing TradingMoE'))
        self.assertIn('missing TradingMoE',s.set_model('candidate',True)['error'])
        self.assertFalse(s.model_enabled['candidate'])
        self.assertFalse(json.loads((s.profile/'agent/autonomy.json').read_text())['candidate_enabled'])

    def test_candidate_operating_start_does_not_start_assembly_trial(self):
        s=self.supervisor;s.assembly_orchestrator=SimpleNamespace(worker=None,trial=Mock())
        s._launch=Mock()
        self.assertTrue(s.set_model('candidate',True)['ok'])
        s.set_model('candidate',False)
        s.assembly_orchestrator.trial.assert_not_called()
        s._launch.assert_called_once_with('candidate')
        self.assertEqual(list(s.model_dir.iterdir()),[])

    def test_trial_does_not_replace_saved_operating_account(self):
        worker=self.workers['candidate'];worker.runner_script='run_assembly_trial.py'
        path=self.supervisor.profile/'agent/candidate_observer_account.json'
        account=PaperAccount(path,.001,.0001)
        account.state['books']['USD']['cash']=8765
        account.state['books']['USD']['positions']={'MSFT':{'quantity':2,'average_cost':100}}
        account.save();before=path.read_bytes()
        with patch('stockrl.provider_credentials.public_status',return_value={'provider':'public','environment':'paper'}):
            status=self.supervisor.status()
        candidate=status['model_runtime']['candidate']
        self.assertEqual(candidate['account_scope'],'long_term')
        self.assertEqual(candidate['books']['USD']['cash'],8765)
        self.assertEqual(candidate['books']['USD']['positions'][0]['symbol'],'MSFT')
        self.assertEqual(path.read_bytes(),before)

    def test_stopped_trial_runner_is_reset_for_candidate_operation(self):
        worker=self.workers['candidate'];worker.runner_script='run_assembly_trial.py'
        s=self.supervisor;s._launch=Mock()
        s.assembly_orchestrator=SimpleNamespace(worker=worker,trial=Mock())
        s.set_model('candidate',True)
        self.assertEqual(worker.runner_script,'run_native_vertical_trading.py')
        self.assertIsNone(s.assembly_orchestrator.worker)
        s.assembly_orchestrator.trial.assert_not_called()

    def test_system_start_only_launches_feed_and_does_not_import_torch(self):
        s=self.supervisor;s.run_requested=False;s._launch=Mock()
        self.assertTrue(s.start('live','1m')['ok'])
        s._launch.assert_called_once_with('feed')
        self.assertFalse(any(s.model_enabled.values()))

    def test_retired_agent_role_cannot_be_launched(self):
        with self.assertRaises(ValueError):self.supervisor._launch('agent')

    def test_status_ignores_stale_transformer_metrics(self):
        s=self.supervisor
        atomic_json({'models':{'champion':{'loaded':True,'status':'running'}},
                     'candidate_training':True,'replay_count':9999},s.profile/'agent/metrics.json')
        with patch('stockrl.provider_credentials.public_status',return_value={'provider':'public','environment':'paper'}):
            status=s.status()
        self.assertFalse(status['agent_process_running'])
        self.assertFalse(status['model_runtime']['champion']['loaded'])
        self.assertEqual(status['learning']['replay_current'],0)
        self.assertFalse(status['metrics']['candidate_training'])

    def test_actual_candidate_worker_drives_health_and_learning(self):
        self.workers['candidate'].status.return_value={'status':'running','alive':True,'pid':123,
            'load_count':1,'learning_active':True,'optimizer_updates':17,
            'learning':{'loss':.25,'samples':4},'replay':{'untrained':2},
            'compute':{'inference_device':'cuda:0','allocated_bytes':128}}
        with patch('stockrl.provider_credentials.public_status',return_value={'provider':'public','environment':'paper'}):
            status=self.supervisor.status()
        self.assertTrue(status['agent_process_running'])
        self.assertEqual(status['agent_health']['candidate']['status'],'healthy')
        self.assertEqual(status['learning']['candidate_optimizer_steps'],17)
        self.assertEqual(status['learning']['replay_current'],2)

    def test_reset_does_not_touch_assembly_trial_accounts(self):
        worker=self.workers['candidate'];worker.runner_script='run_assembly_trial.py'
        trial=worker.state/'assembly/trial/candidate/paper/paper_account.json'
        paper=PaperAccount(trial,.001,.0001);paper.save();before=trial.read_bytes()
        longterm=PaperAccount(worker.state/'paper_account.json',.001,.0001)
        longterm.state['books']['USD']['cash']=9000;longterm.save()
        self.supervisor.fee=.001
        result=self.supervisor.reset_paper_accounts()
        self.assertTrue(result['trial_accounts_preserved'])
        self.assertEqual(trial.read_bytes(),before)
        self.assertEqual(PaperAccount(worker.state/'paper_account.json',.001,.0001).state['books']['USD']['cash'],10000)

    def test_no_retired_transformer_module_or_png_in_repository(self):
        for module in ('global_transformer','global_online','market_training','core','desktop'):
            self.assertFalse((ROOT/'src/stockrl'/(module+'.py')).exists())
        self.assertEqual(list((ROOT/'artifacts/experts/verification').glob('*.png')),[])

if __name__=='__main__':unittest.main()
