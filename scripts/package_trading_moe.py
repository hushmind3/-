"""Pack already verified native weights; no expert probe or download."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from stockrl.expert_registry import read_fusion_output,atomic_json
from stockrl.trading_moe import TradingMoE,package_verified_experts,parameter_digest
from stockrl.paths import TRADING_MOE_CHECKPOINT
from run_trading_moe import all_expert_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--load-only",action="store_true")
    parser.add_argument("--skip-expert-hashes",action="store_true",help="restore without rereading all frozen tensors for hashes")
    args=parser.parse_args()
    target=args.checkpoint or TRADING_MOE_CHECKPOINT
    started=time.perf_counter()
    if args.load_only:
        model,optimizer_state=TradingMoE.load_checkpoint(target)
    else:
        baseline=read_fusion_output(Path("runtime/trading_moe/registry.json"))
        model=package_verified_experts(args.root,baseline,all_expert_snapshot(args.root))
        model.save_checkpoint(target)
    report={"checkpoint":str(target),"bytes":target.stat().st_size,"seconds":time.perf_counter()-started,
        "experts":{k:{"parameters":sum(p.numel() for p in e.parameters()),"hash":None if args.skip_expert_hashes else parameter_digest(e),
                      "frozen":all(not p.requires_grad for p in e.parameters())} for k,e in model.experts.items()},
        "parameters":sum(p.numel() for p in model.parameters()),"expert_count":len(model.experts),
        "controller_parameters":sum(p.numel() for p in model.controller.parameters()),
        "max_gpu_experts":1,"expert_probes_rerun":0,"load_from_one_checkpoint":args.load_only,
        "optimizer_updates":model.optimizer_updates,"controller_hash":parameter_digest(model.controller),
        "optimizer_restored":bool(optimizer_state) if args.load_only else False}
    atomic_json(args.root/"verification"/("TradingMoE.loaded.json" if args.load_only else "TradingMoE.packaged.json"),report)
    print(json.dumps(report))


if __name__=="__main__":main()
