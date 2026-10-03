import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
import random
from unittest.mock import patch

from stockrl.assembly_orchestrator import AssemblyOrchestrator, mutation_key, recipe_fingerprint


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

    def add_expert(self, key="m_new", backend="timesfm", universe=None):
        document=json.loads(self.registry.read_text())
        entry=dict(document["experts"][0],id=key,backend=backend)
        if universe:entry["universe"]=universe
        document["experts"].append(entry)
        self.registry.write_text(json.dumps(document));self.worker.scan_registry()

    def historical_score(self, candidate_id, operations, delta, samples=24, window="same"):
        score={"delta":delta,"source":"real-cache",
            "candidate":{"decisions":samples,"first_as_of":window,"last_as_of":window+"-end"},
            "champion":{"decisions":samples}}
        self.worker._event("rejected",{"candidate_id":candidate_id,"mutation_operations":operations},
            "paired result",scores={"paper":score})

    def test_fingerprint_is_canonical_and_ignores_non_execution_fields(self):
        base=self.worker._candidate_base();other=deepcopy(base)
        other["enabled_experts"].reverse();other["candidate_id"]="different"
        other.update(created_at="other",scores={"paper":2},mutation_description="irrelevant")
        other["refresh_seconds"]["p1"]=999  # Recomputed per bar, not an assembly change.
        other["cache_interval_seconds"]=999
        self.assertEqual(recipe_fingerprint(base),recipe_fingerprint(other))
        other["policy_routing"]["top_k"]=1
        self.assertNotEqual(recipe_fingerprint(base),recipe_fingerprint(other))

    def test_no_duplicate_candidates_across_rejection_and_restart(self):
        self.add_expert("m3");self.add_expert("p3","macrophft")
        self.worker.champion["enabled_experts"].extend(["m3","p3"])
        fingerprints=set()
        for _ in range(35):
            recipe=self.worker.generate()
            self.assertNotIn(recipe["fingerprint"],fingerprints)
            fingerprints.add(recipe["fingerprint"])
        self.worker.settings({"auto_replace":False});self.worker.next()
        restored=AssemblyOrchestrator(directory=self.worker.directory,registry=self.registry,
            checkpoint=self.worker.checkpoint,background=False)
        try:
            for _ in range(10):
                recipe=restored.generate()
                self.assertNotIn(recipe["fingerprint"],fingerprints)
                fingerprints.add(recipe["fingerprint"])
        finally:restored.close()

    def test_new_expert_rejection_does_not_repeat_enable_probe(self):
        self.add_expert()
        recipe=self.worker.generate()
        self.assertEqual(recipe["generation_reason"],"new-expert-probe")
        self.assertEqual(self.worker.state["expert_trials"]["m_new"]["status"],"untested")
        reserved=self.worker.generate()
        self.assertNotEqual(reserved["generation_reason"],"new-expert-probe")
        self.worker.next()
        self.assertEqual(self.worker.state["expert_trials"]["m_new"]["status"],"rejected")
        for _ in range(12):
            later=self.worker.generate()
            self.assertNotEqual(later["fingerprint"],recipe["fingerprint"])
            self.assertNotEqual(later["generation_reason"],"new-expert-probe")

    def test_new_revision_can_be_probed_after_previous_rejection(self):
        self.add_expert();first=self.worker.generate();self.worker.next()
        document=json.loads(self.registry.read_text())
        document["experts"][-1]["files"]=[{"sha256":"updated-real-revision"}]
        self.registry.write_text(json.dumps(document));self.worker.scan_registry()
        changed=self.worker.generate()
        self.assertEqual(changed["generation_reason"],"new-expert-probe")
        self.assertNotEqual(first["fingerprint"],changed["fingerprint"])

    def test_new_expert_qualified_then_promoted_states(self):
        self.add_expert();recipe=self.worker.generate();recipe["evaluation_state"]="replay"
        path=self.worker.directory/"results"/(recipe["candidate_id"]+".json");path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"state":"qualified","scores":{}}))
        self.worker.tick()
        self.assertEqual(self.worker.state["expert_trials"]["m_new"]["status"],"tested")
        self.worker.settings({"auto_promote":True});self.worker.tick()
        self.assertEqual(self.worker.state["expert_trials"]["m_new"]["status"],"promoted")

    def test_policy_mutation_searches_both_router_fields(self):
        pool=self.worker._mutation_pool(self.worker._candidate_base())
        operations=[op for op in pool if op["kind"]=="policy"]
        self.assertEqual({op["field"] for op in operations},
            {"policy_routing.top_k","policy_routing.temperature"})
        generated=[self.worker.generate() for _ in range(25)]
        self.assertTrue(any(r["policy_routing"]["top_k"]!=0 for r in generated))
        self.assertTrue(any(r["policy_routing"]["temperature"]!=1 for r in generated))

    def test_no_effect_stock_toggle_and_policy_refresh_are_excluded(self):
        self.add_expert("stock_only","stock_policy",["MSFT"])
        self.worker.state["evaluation_context"]={"symbols":["ETHUSDT"],
            "market_ages":{"m1":[0,120,600],"m2":[0],"p1":[300]}}
        pool=self.worker._mutation_pool(self.worker._candidate_base())
        self.assertFalse(any(op.get("expert")=="stock_only" for op in pool))
        refresh=[op for op in pool if op["kind"]=="refresh"]
        self.assertTrue(refresh)
        self.assertEqual({op["expert"] for op in refresh},{"m1"})
        self.assertFalse(any(op["value"] in (1800,3600,7200) for op in refresh))

    def test_history_changes_weights_and_preserves_exploration(self):
        good={"kind":"policy","field":"policy_routing.temperature","value":.75}
        bad={"kind":"policy","field":"policy_routing.temperature","value":1.25}
        for i in range(8):
            self.historical_score("good-"+str(i),[good],.004,64,"window-"+str(i))
            self.historical_score("bad-"+str(i),[bad],-.004,64,"window-"+str(i))
        rng=random.Random(14)
        with patch("stockrl.assembly_orchestrator.random.Random",return_value=rng):
            recipe=self.worker.generate()
        weights=recipe["selection_weights"]
        self.assertGreater(weights[mutation_key(good)],weights[mutation_key(bad)])
        self.assertEqual(recipe["exploration_probability"],.25)
        statistics=self.worker.history_stats["mutations"]
        self.assertEqual(statistics[mutation_key(good)]["improved"],8)
        draw=random.Random(42).choices(["good","bad"],weights=[weights[mutation_key(good)],weights[mutation_key(bad)]],k=2000)
        self.assertGreater(draw.count("good"),draw.count("bad")*8)

    def test_same_24_bar_window_does_not_gain_false_confidence(self):
        operation={"kind":"toggle","expert":"m1","enabled":False}
        for i in range(12):self.historical_score("same-"+str(i),[operation],.02,24)
        self.worker._history()
        item=self.worker.history_stats["mutations"][mutation_key(operation)]
        self.assertEqual(item["samples"],24)
        self.assertEqual(item["windows"],1)
        self.assertLess(item["confidence"],.02)
        self.assertLess(item["selection_weight"],1.3)
        self.assertEqual(self.worker.history_stats["experts"]["m1"]["count"],12)

    def test_archive_and_event_scores_count_once_and_legacy_is_loaded(self):
        recipe=self.worker._candidate_base()
        recipe.update(candidate_id="old-recipe",evaluation_state="rejected",mutation_description="m1 OFF",
            enabled_experts=["m2","p1","p2"],scores={"paper":{"delta":-.002,
                "candidate":{"decisions":24},"champion":{"decisions":24}}})
        self.worker.current=recipe;self.worker._archive("legacy test");self.worker.current={}
        self.worker._history()
        operation={"kind":"toggle","expert":"m1","enabled":False}
        self.assertEqual(self.worker.history_stats["mutations"][mutation_key(operation)]["count"],1)
        self.assertIn(recipe_fingerprint(recipe),self.worker._seen_fingerprints)
        for _ in range(15):self.assertNotEqual(self.worker.generate()["fingerprint"],recipe_fingerprint(recipe))

    def test_legacy_router_and_refresh_history_migrates_without_toggle_fields(self):
        for i,text in enumerate(("시장 router top-k=2 / 온도=0.75","m1 refresh=300초")):
            self.worker._event("rejected",{"candidate_id":"legacy-"+str(i),"mutation_description":text},
                "old history",scores={"paper":{"delta":-.001,"candidate":{"decisions":16},"champion":{"decisions":16}}})
        status=self.worker.status()
        self.assertIn("history_stats",status)
        self.assertEqual(len(status["history_stats"]["mutations"]),3)

    def test_successful_mutations_can_be_combined(self):
        first={"kind":"router","field":"market_routing.top_k","value":1}
        second={"kind":"policy","field":"policy_routing.temperature","value":.75}
        self.historical_score("success-one",[first],.001)
        self.historical_score("success-two",[second],.002)
        rng=random.Random(4)
        with patch("stockrl.assembly_orchestrator.random.Random",return_value=rng):
            with patch.object(rng,"random",side_effect=[.8,.1,.5]):
                recipe=self.worker.generate()
        self.assertEqual(recipe["generation_reason"],"successful-mutation-combination")
        self.assertEqual(len(recipe["mutation_operations"]),2)
        self.assertEqual(recipe["market_routing"]["top_k"],1)
        self.assertEqual(recipe["policy_routing"]["temperature"],.75)

    def test_evaluated_fingerprint_in_journal_is_never_generated_again(self):
        base=self.worker._candidate_base()
        operation={"kind":"policy","field":"policy_routing.temperature","value":.75}
        recipe=self.worker._apply_operations(base,[operation])
        recipe["fingerprint"]=recipe_fingerprint(recipe)
        self.worker._event("rejected",recipe,"already evaluated")
        for _ in range(20):self.assertNotEqual(self.worker.generate()["fingerprint"],recipe["fingerprint"])

    def test_history_family_influences_untested_policy_settings(self):
        tested={"kind":"policy","field":"policy_routing.top_k","value":1}
        new={"kind":"policy","field":"policy_routing.temperature","value":.75}
        for i in range(8):self.historical_score("policy-good-"+str(i),[tested],.004,64,str(i))
        self.worker._history()
        self.assertNotIn(mutation_key(new),self.worker.history_stats["mutations"])
        self.assertGreater(self.worker._mutation_weight(new),1.)
        self.assertEqual(self.worker._mutation_weight({"kind":"router","field":"market_routing.top_k","value":1}),1.)

    def test_full_top_k_and_single_policy_temperature_are_not_noop_mutations(self):
        base=self.worker._candidate_base();base["market_routing"]["top_k"]=4
        base["policy_routing"]["top_k"]=1
        pool=self.worker._mutation_pool(base)
        self.assertFalse(any(op.get("field")=="market_routing.top_k" and op["value"]==0 for op in pool))
        self.assertFalse(any(op.get("field")=="policy_routing.temperature" for op in pool))
        combination=self.worker._apply_operations(self.worker._candidate_base(),[
            {"kind":"toggle","expert":"p2","enabled":False},
            {"kind":"policy","field":"policy_routing.temperature","value":.75}])
        self.assertFalse(self.worker._valid_combination(combination))

    def test_exhausted_pool_reports_instead_of_repeating(self):
        with patch.object(self.worker,"_mutation_pool",return_value=[]):
            with self.assertRaisesRegex(ValueError,"같은 후보를 반복하지"):
                self.worker.generate()
        self.assertFalse(self.worker.queue)
        self.assertFalse(self.worker.current)

    def test_legacy_policy_refresh_queue_is_archived_without_resetting_current(self):
        self.worker.generate();current=self.worker.current["candidate_id"]
        legacy=self.worker._candidate_base()
        legacy.update(mutation_description="p1 refresh=3600초",candidate_id="legacy-noop")
        legacy["refresh_seconds"]["p1"]=3600
        self.worker.queue.append(legacy)
        self.worker.generate()
        self.assertEqual(self.worker.current["candidate_id"],current)
        self.assertNotIn("legacy-noop",[r["candidate_id"] for r in self.worker.queue])
        archived=json.loads((self.worker.directory/"recipes/legacy-noop.json").read_text())
        self.assertEqual(archived["evaluation_state"],"rejected")

    def test_policy_routing_changes_actual_controller_forward(self):
        import torch
        from stockrl.trading_moe import VerticalController
        torch.manual_seed(5)
        controller=VerticalController({"m1":8,"m2":8,"macrophft_1":8,"stock":8},["stock"])
        controller.eval()
        controller.policy_router["macrophft_1"].bias.data.fill_(2)
        controller.policy_router["stock"].bias.data.fill_(-1)
        evidence={key:torch.ones(1,1,8) for key in controller.sizes}
        validity={key:torch.ones(1,1,dtype=torch.bool) for key in controller.sizes}
        account=torch.zeros(1,1,16)
        with torch.no_grad():
            controller.assembly_routing={"policy":{"top_k":0,"temperature":1.}}
            baseline=controller(evidence,validity,account)
            controller.assembly_routing={"policy":{"top_k":0,"temperature":.75}}
            colder=controller(evidence,validity,account)
            self.assertFalse(torch.allclose(baseline["policy_router_probabilities"],colder["policy_router_probabilities"]))
            controller.assembly_routing={"policy":{"top_k":1,"temperature":1.}}
            limited=controller(evidence,validity,account)
            self.assertEqual(limited["policy_validity"].sum().item(),1)
            self.assertFalse(torch.allclose(baseline["shared_latent"],limited["shared_latent"]))


if __name__=="__main__":unittest.main()
