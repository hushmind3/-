"""CPU-only contract checks; never touch live accounts or train models."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import psutil

from stockrl.expert_registry import atomic_json, read_registry, read_raw_output, read_fusion_output
from stockrl.expert_system import TradingMoE, ExpertSpec, select_experts, validate_snapshot, pending_head_output, adapter_features, build_fusion_head, registry_owner


class ExpertContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        checkpoint = self.root / "native.bin"
        checkpoint.write_bytes(b"original")
        self.entry = {"id":"timesfm", "backend":"timesfm", "name":"Native model", "role":"forecast",
            "variant":None, "parameters":3, "weight_bytes":12, "dtype":"FP32", "verified":True, "frozen":True,
            "files":[{"path":"native.bin", "bytes":8, "sha256":"0682c5f2076f099c34cfdd15a9e063849ed437a49677e6fcc5b4198c76575be5"}],
            "probe":{"output_shape":[1,1,2], "input_shapes":{"series":[1,128]}, "forward_seconds":.1},
            "router_selected":False}
        import hashlib
        self.entry["files"][0]["sha256"] = hashlib.sha256(b"original").hexdigest()
        self.catalog = {"artifact_root":str(self.root), "experts":[self.entry], "unavailable":[]}
        atomic_json(self.root / "expert_catalog.json", self.catalog)
        self.registry = self.root / "registry.json"

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self):
        return {"symbols":["AAPL"], "as_of":"2026-09-25", "currencies":{"AAPL":"USD"},
            "current_weights":{"AAPL":.25}, "expert_inputs":{}}

    def test_registration_does_not_load(self):
        TradingMoE(self.root, registry_path=self.registry)
        result = read_registry(self.registry)
        self.assertEqual(result["totals"]["parameters"], 3)
        self.assertEqual(result["totals"]["vram_bytes"], 0)
        self.assertFalse(result["experts"][0]["loaded"])

    def test_unverified_rejected(self):
        self.catalog["experts"][0]["verified"] = False
        atomic_json(self.root / "expert_catalog.json", self.catalog)
        with self.assertRaises(ValueError): TradingMoE(self.root, registry_path=self.registry)

    def test_checkpoint_hash_checked(self):
        model = TradingMoE(self.root, registry_path=self.registry)
        model._verify_originals(self.entry)
        (self.root / "native.bin").write_bytes(b"modified")
        with self.assertRaises(ValueError): model._verify_originals(self.entry)

    def test_manifest_roundtrip_and_mismatch(self):
        model = TradingMoE(self.root, registry_path=self.registry)
        manifest = self.root / "manifest.json"
        value = model.save_manifest(manifest)
        TradingMoE.from_manifest(manifest, self.root, registry_path=self.registry)
        value["weights_merged"] = True
        atomic_json(manifest, value)
        with self.assertRaises(ValueError): TradingMoE.from_manifest(manifest, self.root, registry_path=self.registry)

    def test_last_use_survives_wrapper_restart(self):
        model = TradingMoE(self.root, registry_path=self.registry)
        model.registry["experts"][0]["last_used_at"] = "2026-10-02T00:00:00Z"
        model._publish()
        resumed = TradingMoE(self.root, registry_path=self.registry)
        self.assertEqual(resumed.registry["experts"][0]["last_used_at"], "2026-10-02T00:00:00Z")

    def test_router_budget_and_capability(self):
        specs = [ExpertSpec("small", "price", 12, 0), ExpertSpec("other", "price", 12, 1),
                 ExpertSpec("huge", "events", 120, 2)]
        result = select_experts({"small":{}, "other":{}, "huge":{}}, 3, 20, specs)
        self.assertEqual([e.name for e in result], ["small"])

    def test_currency_weights_validation(self):
        data = self.snapshot()
        validate_snapshot(data)
        data["current_weights"]["AAPL"] = 1.1
        with self.assertRaises(ValueError): validate_snapshot(data)

    def test_asof_validation(self):
        data = self.snapshot()
        data["expert_inputs"] = {"timesfm":{"symbols":["AAPL"], "as_of":"tomorrow"}}
        with self.assertRaises(ValueError): validate_snapshot(data)

    def test_untrained_policy_disabled(self):
        value = pending_head_output(self.snapshot())
        self.assertFalse(value["executable"])
        self.assertEqual(value["cash_weights_by_currency"], {"USD":.75})

    def test_dead_worker_cleared(self):
        self.entry["worker"] = {"pid":os.getpid(), "created_at":0, "status_path":"ignored"}
        self.entry.update(loaded=True, active=True)
        atomic_json(self.registry, self.catalog)
        value = read_registry(self.registry)
        self.assertFalse(value["experts"][0]["loaded"])
        self.assertEqual(value["totals"]["ram_bytes"], 0)

    def test_current_memory_not_peak(self):
        sample = self.root / "worker.json"
        atomic_json(sample, {"pid":os.getpid(), "expert_id":"timesfm", "stage":"inference", "device":"cuda:0", "vram_bytes":48})
        self.entry["worker"] = {"pid":os.getpid(), "created_at":psutil.Process().create_time(), "status_path":str(sample)}
        self.entry["last_peak_vram_bytes"] = 100000000
        atomic_json(self.registry, self.catalog)
        value = read_registry(self.registry)
        self.assertEqual(value["totals"]["vram_bytes"], 48)
        self.assertEqual(value["totals"]["active_parameters"], 3)
        self.assertTrue(value["experts"][0]["loaded"])

    def test_raw_output_scope(self):
        raw = self.root / "raw.json"
        packet = {"native_output":[[[1,2]]], "output_shape":[1,1,2]}
        atomic_json(raw, packet)
        self.entry.update(raw_output_path=str(raw), raw_output_origin="independent_verification")
        atomic_json(self.registry, self.catalog)
        self.assertEqual(read_raw_output(self.registry, "timesfm")["packet"], packet)
        with self.assertRaises(StopIteration): read_raw_output(self.registry, "../../native.bin")

    def test_adapter_retains_native_packet(self):
        p = {"expert":"timesfm", "symbols":["AAPL"], "layout":"symbol,horizon,point_and_nine_quantiles",
            "native_output":[[[1,2]]], "units":"daily_excess_return", "horizon":1, "as_of":"date"}
        before = json.dumps(p)
        result = adapter_features([p])
        self.assertEqual(result[0]["features"], [[1.,2.]])
        self.assertEqual(before, json.dumps(p))

    def packet(self, values=None):
        return {"expert":"timesfm", "symbols":["AAPL"], "layout":"symbol,horizon,point_and_nine_quantiles",
            "native_output":values or [[[1,2]]], "units":"daily_excess_return", "horizon":1,
            "sampling_seconds":86400, "as_of":"2026-09-25"}

    def test_adapter_subset_and_mask(self):
        p = self.packet()
        result = adapter_features([p], ["MSFT", "AAPL"])[0]
        self.assertEqual(result["features"], [[0,0],[1,2]])
        self.assertEqual(result["coverage_mask"], [False,True])
        self.assertEqual(result["units"], p["units"])

    def test_adapter_toto_axes(self):
        p = self.packet()
        p.update(layout="nine_quantiles,batch,variate,horizon", native_output=[[[[i]]] for i in range(9)])
        self.assertEqual(adapter_features([p])[0]["features"], [list(range(9))])

    def test_nonfinite_evidence_rejected(self):
        with self.assertRaises(ValueError): adapter_features([self.packet([[[float("nan")]]])])

    def test_subset_snapshot_valid_unknown_rejected(self):
        s = self.snapshot()
        s["expert_inputs"] = {"timesfm":self.packet()}
        validate_snapshot(s)
        s["expert_inputs"]["timesfm"]["symbols"] = ["unknown"]
        with self.assertRaises(ValueError): validate_snapshot(s)

    def test_same_modality_optional_full_sequential_route(self):
        specs = [ExpertSpec("a","price",12,0), ExpertSpec("b","price",12,1)]
        self.assertEqual(len(select_experts({"a":{},"b":{}},2,12,specs,False)),2)

    def test_fusion_really_computes_frozen_outputs(self):
        import torch
        model = TradingMoE(self.root, registry_path=self.registry)
        p = self.packet()
        before = json.dumps(p)
        fusion, trade = model._fuse(adapter_features([p]),self.snapshot())
        self.assertEqual(fusion["shapes"]["shared_latent"],[1,1,64])
        self.assertEqual(fusion["shapes"]["policy_logits"],[1,1,3])
        self.assertFalse(trade["executable"])
        self.assertTrue(all(not param.requires_grad and param.grad is None for param in model.fusion_head.parameters()))
        changed, _ = model._fuse(adapter_features([self.packet([[[10,-20]]])]),self.snapshot())
        self.assertNotEqual(fusion["native_head_output"]["shared_latent"], changed["native_head_output"]["shared_latent"])
        self.assertEqual(before,json.dumps(p))
        self.assertEqual(torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,0)

    def test_fusion_checkpoint_reproduces_output_without_learning(self):
        model = TradingMoE(self.root, registry_path=self.registry)
        a,_ = model._fuse(adapter_features([self.packet()]),self.snapshot())
        restored = TradingMoE(self.root, registry_path=self.registry)
        b,_ = restored._fuse(adapter_features([self.packet()]),self.snapshot())
        self.assertEqual(a,b)
        self.assertFalse(restored.fusion_checkpoint["trained"])

    def test_missing_symbol_mask_retains_held_position_and_currency_budgets(self):
        s = self.snapshot()
        s.update(symbols=["AAPL","MSFT","KR"],currencies={"AAPL":"USD","MSFT":"USD","KR":"KRW"},
                 current_weights={"AAPL":.25,"MSFT":.35,"KR":.4})
        model = TradingMoE(self.root, registry_path=self.registry)
        fusion,trade = model._fuse(adapter_features([self.packet()],s["symbols"]),s)
        self.assertEqual(trade["target_weights"]["MSFT"],.35)
        self.assertEqual(trade["target_weights"]["KR"],.4)
        self.assertEqual(trade["actions"]["KR"],"HOLD")
        self.assertEqual(fusion["native_head_output"]["expert_attention"][0][1],[0.0])
        for c in ("KRW","USD"):
            self.assertAlmostEqual(sum(w for k,w in trade["target_weights"].items() if s["currencies"][k] == c)
                + trade["cash_weights_by_currency"][c],1,places=6)

    def test_untrained_head_requires_explicit_diagnostic_flag(self):
        import torch
        head = build_fusion_head({"a":2}).requires_grad_(False)
        with self.assertRaises(RuntimeError): head({"a":torch.zeros(1,1,2)},torch.zeros(1,1,16))

    def test_wrapper_owner_lock_excludes_concurrent_execution(self):
        owner = self.root / "gpu-owner.lock"
        with patch("stockrl.expert_system.GPU_OWNER_LOCK",owner), registry_owner(owner):
            with self.assertRaises(RuntimeError):
                with registry_owner(owner): self.fail("second owner admitted")
            with self.assertRaises(RuntimeError): TradingMoE(self.root,registry_path=self.root / "other.json")
        with registry_owner(owner): pass

    def test_fusion_output_path_scoped_to_artifact_root(self):
        output = self.root / "result.json"
        atomic_json(output,{"schema":"test"})
        self.catalog["pipeline"] = {"result_path":str(output)}
        atomic_json(self.registry,self.catalog)
        self.assertEqual(read_fusion_output(self.registry),{"schema":"test"})
        self.catalog["pipeline"]["result_path"] = str(self.root.parent / "outside.json")
        atomic_json(self.registry,self.catalog)
        with self.assertRaises(ValueError): read_fusion_output(self.registry)


if __name__ == "__main__": unittest.main()
