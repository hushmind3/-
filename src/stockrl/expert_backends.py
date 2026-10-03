"""Native frozen experts for isolated research; no live agent imports this module.

Run one backend per process so incompatible vendor packages and CUDA residency
cannot leak into another expert. Original checkpoint files are never rewritten.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types
from .paths import expert_weight_path

_source_lock = threading.RLock()


def _source_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def load_native_pretrained(cls, directory, **kwargs):
    """Read config from the project and original tensors from the model folder."""
    import torch
    from safetensors.torch import load_file
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if cls.__name__ == "ChronosPipeline":
        from chronos import ChronosConfig, ChronosModel
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(directory, local_files_only=True))
        state = load_file(str(expert_weight_path(directory / "model.safetensors")))
        # HF safetensors saves shared T5 embeddings once, rather than repeating
        # their aliases. Restore the same tied tensors before strict loading.
        for name in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"):
            if name not in state and (name != "lm_head.weight" or model.config.tie_word_embeddings):
                state[name] = state["shared.weight"]
        model.load_state_dict(state, strict=True)
        model.tie_weights()
        chronos_config = ChronosConfig(**config["chronos_config"])
        return cls(chronos_config.create_tokenizer(), ChronosModel(chronos_config, model))
    model = cls(**config)
    model.load_state_dict(load_file(str(expert_weight_path(directory / "model.safetensors"))), strict=True)
    return model


def source_module(name, path):
    with _source_lock:
        signature = _source_signature(path)
        module = sys.modules.get(name)
        if module is not None and getattr(module, "_stockrl_source_signature", None) == signature:
            return module
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        module._stockrl_source_signature = signature
        return module


def toto_native_module(directory):
    """Execute the original native network, excluding its optional GluonTS bridge.

    The bridge imports Lightning/torchvision even for standalone inference. Only
    that import and that bridge class are omitted; all network AST nodes and
    checkpoint tensors stay unchanged. The original source file is not edited.
    """
    with _source_lock:
        directory = Path(directory)
        path = directory / "model.py"
        signature = (_source_signature(path), _source_signature(directory / "configuration.py"))
        module = sys.modules.get("research_toto.model")
        if module is not None and getattr(module, "_stockrl_source_signature", None) == signature:
            return module
        package = types.ModuleType("research_toto")
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
        source_module("research_toto.configuration", directory / "configuration.py")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gluonts")
            or isinstance(node, ast.ClassDef) and node.name in ("Toto2GluonTSModel", "_FnImputation"))]
        module = types.ModuleType("research_toto.model")
        module.__package__ = "research_toto"
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), "exec"), module.__dict__)
        module._stockrl_source_signature = signature
        return module


def run_native(expert, root, data, device="cpu", status_path=None, expert_id=None):
    run_started = time.perf_counter()
    import numpy as np
    import psutil
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    torch.manual_seed(2026)
    # Compatibility workaround confined to this disposable research process.
    if expert in ("exaone", "chronos", "timemoe"):
        import transformers.utils.import_utils as iu
        iu._torchvision_available = False
    ckpt = root / "checkpoints"
    sources = root / "sources"
    models = []
    extra = {}
    started = time.perf_counter()
    process = psutil.Process()
    rss_start = process.memory_info().rss
    gpu_transfer_seconds = 0.0

    def publish(stage):
        if status_path:
            from .expert_registry import atomic_json, utc_now
            devices = {p.device.type for model in models for p in model.parameters()}
            location = "CPU RAM + GPU VRAM" if devices == {"cpu", "cuda"} else (
                "GPU VRAM" if "cuda" in devices else "CPU RAM" if "cpu" in devices else "disk")
            atomic_json(status_path, {"pid":process.pid, "expert_id":expert_id or expert,
                "stage":stage, "device":device, "updated_at":utc_now(),
                "location":location,
                "vram_bytes":int(torch.cuda.memory_allocated()) if torch.cuda.is_initialized() else 0,
                "vram_reserved_bytes":int(torch.cuda.memory_reserved()) if torch.cuda.is_initialized() else 0})

    publish("loading")

    def frozen(model):
        nonlocal gpu_transfer_seconds
        model.requires_grad_(False).eval()
        models.append(model)
        publish("loading")
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)  # Context startup is part of cold-load.
            transfer_started = time.perf_counter()
            model.to(device)
            torch.cuda.synchronize(device)
            gpu_transfer_seconds += time.perf_counter() - transfer_started
        else:
            model.to(device)
        publish("loading")
        return model

    series = np.asarray(data.get("series", []), dtype=np.float32)
    horizon = int(data.get("horizon", 1))
    if horizon < 1 or horizon > 128:
        raise ValueError("research horizon must be 1..128")
    if expert not in ("kronos", "marketgpt", "macrophft"):
        if series.ndim != 2 or series.shape[1] < 32 or not np.isfinite(series).all():
            raise ValueError("native series must be finite [symbols,time] with >=32 observations")
    if expert in ("chronos", "timesfm") and data.get("units") != "daily_excess_return":
        raise ValueError("FinText requires explicit daily excess returns; raw prices are not equivalent")

    if expert == "chronos":
        from chronos import ChronosPipeline
        pipeline = load_native_pretrained(ChronosPipeline,
            str(ckpt / "Chronos_Small_2023_Global"), device_map="cpu",
            torch_dtype=torch.float32, local_files_only=True,
        )
        frozen(pipeline.model.model)
        def infer():
            samples = pipeline.predict(torch.tensor(series), prediction_length=horizon, num_samples=8)
            return samples.float().cpu().numpy()
        layout = "symbol,sample,horizon"
        units = "daily_excess_return"
    elif expert in ("timesfm", "fincast"):
        if expert == "timesfm":
            ppd = source_module("research_timesfm_decoder", sources / "TimesFM-legacy/src/timesfm/pytorch_patched_decoder.py")
            cfg = ppd.TimesFMConfig(num_layers=9, num_heads=6, num_kv_heads=6,
                                   head_dim=72, hidden_size=432, intermediate_size=1248)
            model = ppd.PatchedTimeSeriesDecoder(cfg)
            state = load_file(str(expert_weight_path(ckpt / "TimesFM_20M_2023_Global/model.safetensors")))
            state = {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}
        else:
            sys.path.insert(0, str(sources / "FinCast-fts/src"))
            ppd = source_module("research_fincast_decoder", sources / "FinCast-fts/src/ffm/pytorch_patched_decoder_MOE.py")
            cfg = ppd.FFMConfig(num_experts=4, gating_top_n=2)
            model = ppd.PatchedTimeSeriesDecoder_MOE(cfg)
            state = torch.load(expert_weight_path(ckpt / "FinCast/v1.pth"), map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
        del state
        frozen(model)
        length = ((min(series.shape[1], 512) + 31) // 32) * 32
        x = np.pad(series[:, -512:], ((0, 0), (length - min(series.shape[1], 512), 0)))
        padding = np.zeros((len(series), length + horizon), np.float32)
        padding[:, :length-min(series.shape[1],512)] = 1
        def infer():
            _, full = model.decode(torch.tensor(x, device=device), torch.tensor(padding, device=device),
                                   torch.full((len(series), 1), int(data.get("frequency_id", 1)), device=device, dtype=torch.long),
                                   horizon_len=horizon)
            return full.float().cpu().numpy()
        layout = "symbol,horizon,point_and_nine_quantiles"
        units = data.get("units", "native_series")
    elif expert == "exaone":
        sys.path.insert(0, str(sources / "EXAONE-Forecast/src"))
        from exaone_forecast.finance.config import EXAONEFinanceConfig
        from exaone_forecast.finance.model import EXAONEFinance
        from exaone_forecast.finance.forecaster import EXAONEFinanceForecaster
        directory = ckpt / "EXAONE-Forecast-for-Finance-1.0"
        cfg = EXAONEFinanceConfig.from_pretrained(str(directory), local_files_only=True)
        model = EXAONEFinance(cfg)
        model.load_state_dict(load_file(str(expert_weight_path(directory / "exaone-finance-1.0.safetensors"))), strict=True)
        frozen(model)
        # Official API loader expects model.safetensors, while the official HF
        # artifact has another name. Attach the verbatim loaded model directly.
        forecaster = EXAONEFinanceForecaster.__new__(EXAONEFinanceForecaster)
        forecaster.model = model
        forecaster.device = device
        forecaster.context_length = int(model.forecasting_config.context_length)
        forecaster._patch = int(model.forecasting_config.output_patch_size)
        forecaster._max_op = int(model.forecasting_config.max_output_patches)
        forecaster.quantiles = [round(float(q), 4) for q in model.quantiles.cpu().tolist()]
        forecaster.max_horizon = forecaster._patch * forecaster._max_op
        def infer():
            return forecaster.predict(series, horizon=horizon, batch_size=len(series))
        layout = "symbol,21_quantiles,horizon"
        units = data.get("units", "native_series")
    elif expert == "timemoe":
        directory = ckpt / "TimeMoE-200M"
        package = types.ModuleType("research_timemoe")
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
        source_module("research_timemoe.configuration_time_moe", directory / "configuration_time_moe.py")
        source_module("research_timemoe.ts_generation_mixin", directory / "ts_generation_mixin.py")
        module = source_module("research_timemoe.modeling_time_moe", directory / "modeling_time_moe.py")
        cfg = module.TimeMoeConfig.from_pretrained(str(directory), local_files_only=True)
        model = module.TimeMoeForPrediction(cfg).to(dtype=torch.bfloat16)
        model.load_state_dict(load_file(str(expert_weight_path(directory / "model.safetensors"))), strict=True)
        frozen(model)
        mean = series.mean(axis=1, keepdims=True)
        scale = series.std(axis=1, keepdims=True).clip(1e-6)
        def infer():
            normalized = torch.tensor((series - mean) / scale, device=device, dtype=torch.bfloat16)
            # Native one-step outputs, iterated without HF's version-dependent
            # generation-cache helper. This does not change model parameters.
            predictions = []
            for _ in range(horizon):
                forecast = model(input_ids=normalized[..., None], use_cache=False,
                    max_horizon_length=1, return_dict=True).logits[:, -1, 0]
                predictions.append(forecast)
                normalized = torch.cat([normalized, forecast[:, None]], dim=-1)
            return torch.stack(predictions, dim=-1).float().cpu().numpy() * scale + mean
        layout = "symbol,horizon"
        units = data.get("units", "native_series")
    elif expert == "toto":
        native = toto_native_module(sources / "toto/toto2/toto2")
        directory = ckpt / "Toto-2.0-313m"
        cfg = native.Toto2ModelConfig(**json.loads((directory / "config.json").read_text()))
        model = native.Toto2Model(cfg)
        model.load_state_dict(load_file(str(expert_weight_path(directory / "model.safetensors"))), strict=True)
        frozen(model)
        def infer():
            x = torch.tensor(series[None], device=device)
            return model.forecast({"target": x, "target_mask": torch.ones_like(x, dtype=torch.bool),
                "series_ids": torch.zeros((1,len(series)),device=device,dtype=torch.long)},
                horizon=horizon, has_missing_values=False).float().cpu().numpy()
        layout = "nine_quantiles,batch,variate,horizon"
        units = data.get("units", "native_series")
    elif expert == "kronos":
        import pandas as pd
        sys.path.insert(0, str(sources / "Kronos"))
        from model import Kronos, KronosTokenizer, KronosPredictor
        model = load_native_pretrained(Kronos, str(ckpt / "Kronos-base"), local_files_only=True)
        tokenizer = load_native_pretrained(KronosTokenizer, str(ckpt / "Kronos-Tokenizer-base"), local_files_only=True)
        frozen(model); frozen(tokenizer)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        def infer():
            outputs = []
            for bars in data["bars"]:
                frame = pd.DataFrame(bars)
                required = ["open", "high", "low", "close", "volume", "amount"]
                if not set(required + ["timestamp"]).issubset(frame.columns):
                    raise ValueError("Kronos needs explicit OHLCV/amount and timestamps")
                if not np.isfinite(frame[required].to_numpy(float)).all():
                    raise ValueError("Kronos OHLCV/amount must be finite")
                stamps = pd.to_datetime(frame.timestamp)
                future = pd.Series(pd.to_datetime(data["future_timestamps"]))
                forecast = predictor.predict(frame[required], stamps, future, horizon, sample_count=1, verbose=False)
                outputs.append(forecast[required].to_numpy(float))
            return np.asarray(outputs)
        layout = "symbol,horizon,OHLCV_amount"
        units = "native_OHLCV_amount"
    elif expert == "marketgpt":
        module = source_module("research_marketgpt", sources / "MarketGPT/equities/fast_model.py")
        path = ckpt / "MarketGPT-100m/ckpt_finetune_AAPL_v3.pt"
        saved = torch.load(expert_weight_path(path), map_location="cpu", weights_only=True, mmap=True)
        args = dict(saved["model_args"])
        tokens = np.asarray(data["itch_tokens"], dtype=np.int64)
        if data.get("token_schema") != "MarketGPT_ITCH_Vocab_v3" or tokens.ndim != 2:
            raise ValueError("MarketGPT requires authentic native encoded ITCH token batches")
        if tokens.min() < 0 or tokens.max() >= args["vocab_size"]:
            raise ValueError("MarketGPT token outside the native vocabulary")
        args["max_seq_len"] = max(32, tokens.shape[1])
        args["dropout"] = 0.0
        model = module.Transformer(module.ModelArgs(**args))
        # Original checkpoint includes >5GB repeated deterministic causal masks.
        # Regenerate ONLY these buffers at the supplied context length; retain
        # every learned parameter and the source artifact byte-for-byte.
        state = {k: v for k,v in saved["model"].items() if not k.endswith(".attention.mask")}
        loaded = model.load_state_dict(state, strict=False)
        if loaded.unexpected_keys or any(not k.endswith(".attention.mask") for k in loaded.missing_keys):
            raise ValueError(f"unexpected MarketGPT schema: {loaded}")
        del saved, state
        frozen(model)
        def infer():
            logits = model(torch.tensor(tokens, device=device)).float()
            return logits.cpu().numpy()
        layout = "batch,last_token,vocabulary"
        units = "next_message_token_logits"
        extra["input_authenticity"] = data.get("input_authenticity", "caller_asserted")
    elif expert == "macrophft":
        directory = sources / "MacroHFT"
        module = source_module("research_macrophft", directory / "model/net.py")
        variant = data.get("variant", "slope/1")
        if variant not in [f"{regime}/{label}" for regime in ("slope","vol") for label in (1,2,3)]:
            raise ValueError("unknown MacroHFT sub-agent")
        model = module.subagent(36, 9, 2, 64)
        model.load_state_dict(torch.load(expert_weight_path(directory / f"result/low_level/ETHUSDT/best_model/{variant}/best_model.pkl"),
                                       map_location="cpu", weights_only=True), strict=True)
        frozen(model)
        def infer():
            return model(torch.tensor(data["single_state"], device=device, dtype=torch.float32),
                         torch.tensor(data["trend_state"], device=device, dtype=torch.float32),
                         torch.tensor(data["previous_action"], device=device, dtype=torch.long)).float().cpu().numpy()
        layout = "batch,Q_flat_long"
        units = "native_ETHUSDT_policy_Q"
        extra["variant"] = variant
        extra["input_authenticity"] = data.get("input_authenticity", "caller_asserted")
    else:
        raise ValueError(f"no verified pretrained backend: {expert}")

    loaded_seconds = time.perf_counter() - started
    cold_load_seconds = time.perf_counter() - run_started - gpu_transfer_seconds
    publish("loaded")
    rss_loaded = process.memory_info().rss
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = [(p, p._version) for model in models for p in model.parameters()]
    publish("inference")
    forward_start = time.perf_counter()
    with torch.inference_mode():
        output = np.asarray(infer())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_start
    if not np.isfinite(output).all():
        raise ValueError("expert returned nonfinite values")
    if any(p.requires_grad or p._version != version for p, version in before):
        raise ValueError("frozen expert parameter was mutated")
    publish("completed")
    memory = process.memory_info()
    shapes = {key:list(np.asarray(data[key]).shape) for key in
              ("series", "itch_tokens", "single_state", "trend_state", "previous_action") if key in data}
    if "bars" in data:
        shapes["OHLCV_amount"] = [len(data["bars"]), len(data["bars"][0]), 6]
    return {"expert":expert, "native_output":output.tolist(), "layout":layout,
        "units":units, "symbols":data.get("symbols", []), "as_of":data.get("as_of"),
        "input_authenticity":data.get("input_authenticity"), "amount_observed":data.get("amount_observed"),
        "horizon":horizon, "sampling_seconds":data.get("sampling_seconds"),
        "parameters":sum(p.numel() for model in models for p in model.parameters()),
        "native_modules":[{"type":type(model).__name__,
            "parameters":sum(p.numel() for p in model.parameters()),
            "buffer_elements":sum(b.numel() for b in model.buffers()),
            "buffer_bytes":sum(b.numel()*b.element_size() for b in model.buffers())} for model in models],
        "parameter_bytes":sum(p.numel()*p.element_size() for model in models for p in model.parameters()),
        "parameter_dtypes":sorted({str(p.dtype) for model in models for p in model.parameters()}),
        "frozen":True, "output_shape":list(output.shape), "load_seconds":loaded_seconds,
        "input_shapes":shapes, "rss_start_bytes":rss_start, "rss_loaded_bytes":rss_loaded,
        "rss_end_bytes":memory.rss, "peak_ram_bytes":getattr(memory,"peak_wset", max(rss_start,rss_loaded,memory.rss)),
        "ram_measurement":"Windows process peak working set" if hasattr(memory,"peak_wset") else "sampled RSS lower bound",
        "forward_seconds":forward_seconds,
        "cold_load_seconds":cold_load_seconds, "gpu_transfer_seconds":gpu_transfer_seconds,
        "worker_seconds":time.perf_counter()-run_started,
        "peak_allocated_bytes":int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None,
        "peak_reserved_bytes":int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else None,
        **extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expert")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--status", type=Path)
    parser.add_argument("--expert-id")
    args = parser.parse_args()
    result = run_native(args.expert, args.root, json.loads(args.input.read_text(encoding="utf-8")), args.device,
                        args.status, args.expert_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "native_output"}))


if __name__ == "__main__":
    main()
