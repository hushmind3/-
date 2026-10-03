import json
from pathlib import Path
import tempfile
import unittest

from stockrl.assembly_orchestrator import AssemblyOrchestrator


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.registry=self.root/"registry.json"
        entries=[{"id":key,"backend":backend,"verified":True,
                  "probe":{"input_shapes":{"input":[1,36]}},"files":[{"sha256":key}]}
                 for key,backend in (("m1","timesfm"),("m2","kronos"),("p1","macrophft"),("p2","macrophft"))]
        self.registry.write_text(json.dumps({"experts":entries}))
        checkpoint=self.root/"shared.pt";checkpoint.write_bytes(b"test fixture, never loaded")
        self.worker=AssemblyOrchestrator(directory=self.root/"assembly",registry=self.registry,checkpoint=checkpoint,background=False)
        self.worker._base_identity()

    def tearDown(self):
        self.worker.close();self.temp.cleanup()

    def test_mutations_keep_native_roles_and_inputs(self):
        for _ in range(12):self.worker.generate()
        for recipe in [self.worker.current]+self.worker.queue:
            self.assertTrue(any(recipe["expert_roles"][key]=="market" for key in recipe["enabled_experts"]))
            self.assertTrue(any(recipe["expert_roles"][key]=="policy" for key in recipe["enabled_experts"]))
            self.assertEqual(recipe["symbol_applicability"]["p1"],["ETHUSDT"])
            self.assertEqual(recipe["native_inputs"]["p1"],{"input":[1,36]})
            self.assertEqual(recipe["base_checkpoint_hash"],self.worker.state["base_hash"])
        self.assertEqual(list((self.root/"assembly").rglob("*.pt")),[])

    def test_rejection_installs_next_and_survives_restart(self):
        self.worker.generate();first=self.worker.current["candidate_id"]
        self.worker.generate();queued=self.worker.queue[0]["candidate_id"]
        self.worker.next()
        self.assertEqual(self.worker.current["candidate_id"],queued)
        self.assertEqual(self.worker.state["rejections"],1)
        restored=AssemblyOrchestrator(directory=self.worker.directory,registry=self.registry,checkpoint=self.worker.checkpoint,background=False)
        self.assertEqual(restored.current["candidate_id"],queued)
        self.assertTrue((self.worker.directory/"recipes"/(first+".json")).exists())
        restored.close()

    def test_registry_change_is_detected_and_enters_generation(self):
        document=json.loads(self.registry.read_text())
        new=dict(document["experts"][0],id="m_new")
        document["experts"].append(new);self.registry.write_text(json.dumps(document))
        self.worker.scan_registry()
        self.assertIn("m_new",self.worker.state["new_experts"])
        self.worker.generate()
        self.assertIn("m_new",self.worker.current["enabled_experts"])
        self.assertEqual(len(self.worker.experts),5)

    def test_detection_and_replace_switches_are_independent(self):
        self.worker.settings({"detect_experts":False,"auto_replace":False})
        document=json.loads(self.registry.read_text());document["experts"].pop()
        self.registry.write_text(json.dumps(document));self.worker.scan_registry()
        self.assertEqual(len(self.worker.experts),4)
        self.worker.generate();self.worker.next()
        self.assertFalse(self.worker.current)
        self.assertFalse(self.worker.state["enabled"])
        with self.assertRaises(ValueError):self.worker.settings({"auto_promote":"yes"})

    def write_trial_result(self, state):
        self.worker.generate();self.worker.generate()
        self.worker.current["evaluation_state"]="replay"
        candidate_id=self.worker.current["candidate_id"]
        result=self.worker.directory/"results"/(candidate_id+".json")
        result.parent.mkdir(exist_ok=True)
        result.write_text(json.dumps({"candidate_id":candidate_id,"state":state,"reason":"comparison test","scores":{}}))
        return candidate_id

    def test_finished_rejection_rotates_without_starting_automation(self):
        rejected=self.write_trial_result("rejected")
        queued=self.worker.queue[0]["candidate_id"]
        self.worker.tick()
        self.assertFalse(self.worker.state["enabled"])
        self.assertEqual(self.worker.current["candidate_id"],queued)
        self.assertNotEqual(rejected,queued)
        self.assertEqual(self.worker.state["rejections"],1)
        self.assertEqual(self.worker.current["evaluation_state"],"ready")

    def test_qualified_candidate_waits_for_promotion_switch(self):
        qualified=self.write_trial_result("qualified")
        self.worker.tick()
        self.assertEqual(self.worker.current["evaluation_state"],"qualified")
        self.assertEqual(self.worker.state["promotions"],0)
        self.worker.settings({"auto_promote":True});self.worker.tick()
        self.assertEqual(self.worker.champion["candidate_id"],qualified)
        self.assertEqual(self.worker.state["promotions"],1)


if __name__=="__main__":unittest.main()
