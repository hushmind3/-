"""CPU-only contract checks; never touch live accounts or train models."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import psutil

from stockrl.expert_registry import atomic_json, read_registry, read_raw_output
from stockrl.expert_system import TradingMoE, ExpertSpec, select_experts, validate_snapshot, pending_head_output, adapter_features


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


if __name__ == "__main__": unittest.main()
