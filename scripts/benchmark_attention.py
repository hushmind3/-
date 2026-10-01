"""Short paired MHA/SDPA benchmark at the live transformer's tensor dimensions."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch


def run(args):
    from stockrl.global_transformer import TransformerBlock, TransformerConfig
    result = {"device": "cuda" if torch.cuda.is_available() else "cpu", "cases": []}
    device = torch.device(result["device"])
    if device.type != "cuda":
        result.update(status="skipped", reason="CUDA is unavailable")
        return result
    free, total = torch.cuda.mem_get_info(device)
    result["free_vram_before_bytes"] = free
    if free < 2 * 1024**3:
        result.update(status="skipped", reason="Less than 2 GiB free VRAM; live inference has priority")
        return result
    cfg = TransformerConfig(d_model=1408, n_heads=16, n_layers=21, ff_mult=4, dropout=0.0)
    for label, batch, length in (("temporal_172x128", 172, 128), ("cross_128x172", 128, 172)):
        for training in (True, False):
            block = TransformerBlock(cfg).to(device=device, dtype=torch.float16)
            x = torch.randn(batch, length, cfg.d_model, device=device, dtype=torch.float16, requires_grad=training)
            mask = torch.zeros(batch, length, device=device, dtype=torch.bool)
            mask[:, 0] = True

            def evaluate(enabled, capture=False):
                block._stockrl_sdpa_enabled = enabled
                block.train(training)
                block.zero_grad(set_to_none=True)
                if training: x.grad = None
                with torch.set_grad_enabled(training):
                    y = block(x, mask)
                    if training:
                        y.float().square().mean().backward()
                torch.cuda.synchronize(device)
                if capture:
                    return y.detach().cpu(), None if not training else x.grad.detach().cpu(), \
                        None if not training else next(block.parameters()).grad.detach().cpu()
                return

            old = evaluate(False, True)
            new = evaluate(True, True)
            output_error = (old[0].float()-new[0].float()).abs().max().item()
            input_grad_error = (old[1].float()-new[1].float()).abs().max().item() if training else 0.0
            weight_grad_error = (old[2].float()-new[2].float()).abs().max().item() if training else 0.0
            if not all(torch.isfinite(v).all().item() for v in new if v is not None):
                raise RuntimeError("optimized attention produced a non-finite output/gradient")
            times = {"mha": [], "sdpa": []}
            for backend in ("mha", "sdpa", "sdpa", "mha", "mha", "sdpa"):
                for _ in range(1):
                    torch.cuda.synchronize(device); start = time.perf_counter()
                    evaluate(backend == "sdpa")
                    times[backend].append(time.perf_counter()-start)
            mha = statistics.median(times["mha"]); sdpa = statistics.median(times["sdpa"])
            result["cases"].append({"shape": label, "mode": "train" if training else "inference",
                "median_mha_seconds": mha, "median_sdpa_seconds": sdpa,
                "speedup": mha/sdpa if sdpa else None,
                "max_output_abs_error": output_error,
                "max_input_gradient_abs_error": input_grad_error,
                "max_weight_gradient_abs_error": weight_grad_error})
            del old, new, x, block
            torch.cuda.empty_cache()
    train = [case for case in result["cases"] if case["mode"] == "train"]
    infer = [case for case in result["cases"] if case["mode"] == "inference"]
    result.update(status="complete", training_speedup_geomean=statistics.geometric_mean(c["speedup"] for c in train),
                  inference_speedup_geomean=statistics.geometric_mean(c["speedup"] for c in infer),
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
        report = run(args)
    except BaseException as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    if report["status"] == "failed":
        raise SystemExit(1)
