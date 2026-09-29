from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import (Config, TemporalActorCritic, device_for, evaluate, infer, load_checkpoint,
                   load_market, make_teachers, observations, ppo_finetune, save_checkpoint,
                   seed_all, split_indices, teacher_targets, train_distill)
from .continuous import run_continuous, run_replay
from .global_transformer import GlobalMarketPanel, GlobalMarketTransformer, TransformerConfig, parameter_count
from .global_online import OnlineGlobalAgent, benchmark_model, load_model
from .paths import default_runtime_dir


def train(args: argparse.Namespace) -> None:
    cfg=Config(window=args.window,fee=args.fee,seed=args.seed,device=args.device,epochs=args.epochs,
               batch_size=args.batch_size,ppo_updates=args.ppo_updates)
    seed_all(cfg.seed); device=device_for(cfg.device); print(f"device={device}")
    df=load_market(args.data)
    train_ix,valid_ix,test_ix=split_indices(len(df),cfg.train_fraction,cfg.valid_fraction)
    x=observations(df,cfg.window); teachers=make_teachers(json.loads(Path(args.teachers).read_text(encoding="utf-8")))
    print("teacher_weights="+json.dumps({t.name:t.weight for t in teachers}))
    # Only request teacher labels for training timestamps; later split rows never
    # enter the distilled training target arrays.
    targets=np.zeros((len(df),3),np.float32); values=np.zeros(len(df),np.float32)
    train_targets,train_values=teacher_targets(df.iloc[train_ix].reset_index(drop=True),teachers,cfg.window)
    targets[train_ix]=train_targets; values[train_ix]=train_values
    model=TemporalActorCritic(d_model=cfg.d_model).to(device)
    ckpt_dir=Path(args.checkpoint_dir)
    train_distill(model,x,targets,values,train_ix,cfg,device,ckpt_dir)
    val=evaluate(model,df,x,valid_ix,cfg,device); print("validation="+json.dumps(val))
    ppo_finetune(model,df,x,train_ix,cfg,device,ckpt_dir)
    val=evaluate(model,df,x,valid_ix,cfg,device); test=evaluate(model,df,x,test_ix,cfg,device)
    print("validation_after_ppo="+json.dumps(val)); print("test_backtest="+json.dumps(test))
    save_checkpoint(ckpt_dir/"final.pt",model,cfg)
    print(f"saved={ckpt_dir/'final.pt'}")
    # Re-open the final artifact and run the inference path as part of the CLI workflow.
    reloaded,_=load_checkpoint(ckpt_dir/"final.pt",device)
    print("sample_inference="+json.dumps(infer(reloaded,df,cfg.window,device)))


def predict(args: argparse.Namespace) -> None:
    device=device_for(args.device); model,cfg=load_checkpoint(args.checkpoint,device)
    print(json.dumps(infer(model,load_market(args.data),cfg.window,device),indent=2))


def public_teacher_dataset(args: argparse.Namespace) -> None:
    from .public_teachers import _logits_for
    df=load_market(args.data); spec=json.loads(Path(args.teachers).read_text(encoding="utf-8")); teachers=make_teachers(spec)
    target=Path(args.output); target.mkdir(parents=True,exist_ok=True)
    all_probs=[]; all_vals=[]; w=np.asarray([max(t.weight,0) for t in teachers],np.float64); w/=w.sum()
    for t in teachers:
        if hasattr(t,"predict_series"):
            logits,values=t.predict_series(df,int(getattr(t,"required_window",args.window)))
        else:
            logits,values=teacher_targets(df,[t],args.window); logits=np.asarray(logits); values=np.asarray(values)
        p=torch.softmax(torch.as_tensor(logits),-1).numpy(); all_probs.append(p); all_vals.append(values)
        pred=pd.DataFrame({"date":df.date.astype(str),"action":[("SELL","HOLD","BUY")[i] for i in p.argmax(-1)],
                           "p_sell":p[:,0],"p_hold":p[:,1],"p_buy":p[:,2],"value":values})
        pred.to_csv(target/f"{t.name}.csv",index=False)
        print(f"teacher={t.name} rows={len(pred)} file={target/(t.name+'.csv')}")
    ens=np.average(np.stack(all_probs),axis=0,weights=w); vals=np.average(np.stack(all_vals),axis=0,weights=w)
    pd.DataFrame({"date":df.date.astype(str),"action":[("SELL","HOLD","BUY")[i] for i in ens.argmax(-1)],
                  "p_sell":ens[:,0],"p_hold":ens[:,1],"p_buy":ens[:,2],"value":vals}).to_csv(target/"ensemble.csv",index=False)
    np.savez_compressed(target/"distillation_targets.npz",probabilities=ens.astype(np.float32),values=vals.astype(np.float32))


def global_info(args: argparse.Namespace) -> None:
    import psutil
    from .core import device_for
    cfg=TransformerConfig(); model=GlobalMarketTransformer(cfg)
    params=parameter_count(model); bytes_fp32=params*4
    info={"architecture":"alternating temporal/cross-market Transformer actor-critic",
          "config":cfg.__dict__,"parameters":params,"fp32_weight_bytes":bytes_fp32,
          "fp16_weight_bytes":params*2,"device":str(device_for(args.device)),
          "process_rss_bytes":psutil.Process().memory_info().rss}
    if args.data:
        panel=GlobalMarketPanel(args.data); info.update({"symbols":len(panel.symbols),"dates":len(panel.dates),
          "date_min":str(panel.dates[0]),"date_max":str(panel.dates[-1]),"benchmark":benchmark_model(model,panel,args.window,args.repeats)})
    print(json.dumps(info,indent=2))


def global_online(args: argparse.Namespace) -> None:
    import psutil
    cfg=TransformerConfig()
    agent=OnlineGlobalAgent(args.state_dir,args.device,cfg,capacity=args.replay_capacity,window=args.window,
      horizon=args.horizon,fee=args.fee,slippage_bps=args.slippage_bps,min_replay=args.min_replay,
      batch_size=args.batch_size,updates_per_candidate=args.updates,lr=args.lr,seed=args.seed,
      candidate_interval=args.candidate_every,initial_champion=args.initial_champion,
      teacher_replay_path=args.teacher_replay,model_dir=args.model_dir)
    try:
        if args.follow and not args.teacher_decisions:
            print(f"following={args.data} poll_seconds={args.poll_seconds}; stop with Ctrl+C")
            agent.follow_csv(args.data,args.poll_seconds,args.initial_lookback_bars)
            return
        panel=GlobalMarketPanel(args.data,max_symbols=agent.cfg.max_symbols,
            symbol_map=getattr(agent.champion,"_stockrl_symbol_map",None),
            recent_timestamps=512 if args.follow else None,
            active_stale_seconds=300 if args.follow else None)
        if args.teacher_decisions:
            teacher_end=None if args.follow else max(2,int(len(panel.dates)*.70))
            print(f"teacher_imitation_rows_added={agent.import_teacher_csv(panel,args.teacher_decisions,args.teacher_symbol,teacher_end,args.teacher_source)}")
        if args.follow:
            print(f"following={args.data} poll_seconds={args.poll_seconds}; stop with Ctrl+C")
            agent.follow_csv(args.data,args.poll_seconds,args.initial_lookback_bars)
            return
        # Measure the unmodified champion first on the final chronological 15%;
        # the same sealed timestamps are measured again after candidate updates.
        test_start=max(max(2,int(len(panel.dates)*.70))+1,int(len(panel.dates)*.85))
        pre_training_test=agent.evaluate_panel(panel,agent.champion,start_index=test_start,stride=args.stride)
        decisions=agent.run_panel(panel,max_observations=args.max_observations,stride=args.stride)
        # Drain one final candidate update after all replay items mature.
        if (agent.replay.trainable_count()>=args.min_replay and
            len({item[7] for item in agent.validation})>=agent.min_validation_dates):
            if not agent.thread.is_alive(): agent.start()
            deadline=__import__("time").time()+args.wait_seconds
            while agent.candidate is not None or agent.metrics["updates"]==0:
                if __import__("time").time()>deadline: break
                __import__("time").sleep(.2)
                if not agent.thread.is_alive() and not agent.stop.is_set(): agent.start()
                if agent.metrics["updates"] and agent.candidate is None: break
        elif agent.replay.trainable_count()>=args.min_replay:
            agent.metrics["candidate_skip_reason"]="held-out validation window has too few timestamps"
        post_training_test=agent.evaluate_panel(panel,agent.champion,start_index=test_start,stride=args.stride)
        candidate_path=agent.model_dir/"candidate.pt"
        candidate_test=None
        if candidate_path.is_file():
            candidate_model,_=load_model(candidate_path,agent.device)
            candidate_test=agent.evaluate_panel(panel,candidate_model,start_index=test_start,stride=args.stride)
            del candidate_model
        backtest=agent.backtest_panel(panel,args.stride)
        agent.checkpoint()
        # Exercise checkpoint reload and confirm same architecture/parameter count.
        reloaded,loaded_cfg=load_model(agent.champion_path,agent.device)
        reload_args=[x.to(agent.device) for x in panel.window(min(len(panel.dates)-1,args.window),args.window)]
        reload_args[0]=reload_args[0].to(dtype=next(reloaded.parameters()).dtype)
        logits,_=reloaded(*reload_args)
        metrics=json.loads((agent.state_dir/"metrics.json").read_text(encoding="utf-8"))
        metrics.update({"market_file":args.data,"market_symbols":len(panel.symbols),"market_dates":len(panel.dates),
          "decisions_file":str(agent.state_dir/"decisions.csv"),"decision_rows":len(decisions),
          "checkpoint_reload":"ok","reload_logits_shape":list(logits.shape),"reload_parameters":parameter_count(reloaded),
          "backtest":backtest,"pre_training_test":pre_training_test,"post_training_test":post_training_test,
          "candidate_test":candidate_test,
          "test_net_return_change":post_training_test["net_return"]-pre_training_test["net_return"],
          "test_net_pnl_return_sum_change":post_training_test["net_pnl_return_sum"]-pre_training_test["net_pnl_return_sum"],
          "candidate_test_net_pnl_return_sum_change":(candidate_test["net_pnl_return_sum"]-pre_training_test["net_pnl_return_sum"]
              if candidate_test is not None else None),
          "config":loaded_cfg.__dict__,"process_peak_rss_bytes":psutil.Process().memory_info().rss,
          "weight_changed":bool(any(x>0 for x in agent.metrics["weight_delta_l1"]))})
        (agent.state_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        print(json.dumps(metrics,indent=2))
    finally:
        agent.close()


def main() -> None:
    p=argparse.ArgumentParser(prog="stockrl",description="Distill weighted trading teachers into a temporal PyTorch actor-critic")
    sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train",help="distill teachers, PPO fine-tune, validate and backtest")
    t.add_argument("--data",default="data/sample_ohlcv.csv"); t.add_argument("--teachers",default="configs/teachers.json")
    t.add_argument("--checkpoint-dir",default=str(default_runtime_dir()/"baseline-checkpoints")); t.add_argument("--window",type=int,default=32)
    t.add_argument("--fee",type=float,default=.001); t.add_argument("--seed",type=int,default=7)
    t.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    t.add_argument("--epochs",type=int,default=15); t.add_argument("--batch-size",type=int,default=128)
    t.add_argument("--ppo-updates",type=int,default=4); t.set_defaults(func=train)
    q=sub.add_parser("predict",help="load a trained checkpoint and infer the latest action")
    q.add_argument("--data",required=True); q.add_argument("--checkpoint",default=str(default_runtime_dir()/"baseline-checkpoints"/"final.pt"))
    q.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"]); q.set_defaults(func=predict)
    d=sub.add_parser("teacher-dataset",help="run configured trained public teachers and save per-model plus ensemble targets")
    d.add_argument("--data",required=True); d.add_argument("--teachers",default="configs/teachers.json")
    d.add_argument("--output",default=str(default_runtime_dir()/"public-teachers")); d.add_argument("--window",type=int,default=32)
    d.set_defaults(func=public_teacher_dataset)
    c=sub.add_parser("continuous",help="observe an updating CSV and paper trade/learn until stopped")
    c.add_argument("--data",required=True,help="append/update-only OHLCV CSV feed")
    c.add_argument("--feedback",default=None,help="optional execution feedback CSV: timestamp,action,reward")
    c.add_argument("--champion",default=str(default_runtime_dir()/"baseline-checkpoints"/"final.pt")); c.add_argument("--state-dir",default=str(default_runtime_dir()/"continuous"))
    c.add_argument("--window",type=int,default=32); c.add_argument("--fee",type=float,default=.001)
    c.add_argument("--horizon",type=int,default=5); c.add_argument("--poll-seconds",type=float,default=10)
    c.add_argument("--validation-rows",type=int,default=64); c.add_argument("--minimum-delta",type=float,default=0.0)
    c.add_argument("--min-replay",type=int,default=32); c.add_argument("--replay-capacity",type=int,default=10000)
    c.add_argument("--seed",type=int,default=7); c.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    c.add_argument("--once",action="store_true",help="process current feed once for verification")
    c.set_defaults(func=run_continuous)
    r=sub.add_parser("replay",help="replay historical rows through the paper decision/outcome loop")
    r.add_argument("--data",required=True); r.add_argument("--checkpoint",default=str(default_runtime_dir()/"baseline-checkpoints"/"final.pt"))
    r.add_argument("--state-dir",default=str(default_runtime_dir()/"replay")); r.add_argument("--window",type=int,default=32)
    r.add_argument("--fee",type=float,default=.001); r.add_argument("--horizon",type=int,default=5)
    r.add_argument("--stride",type=int,default=1); r.add_argument("--replay-capacity",type=int,default=10000)
    r.add_argument("--seed",type=int,default=7); r.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    r.set_defaults(func=run_replay)
    gi=sub.add_parser("global-info",help="report the fixed 0.5B global Transformer size and optional real-panel inference timing")
    gi.add_argument("--data",default=None); gi.add_argument("--window",type=int,default=8); gi.add_argument("--repeats",type=int,default=3)
    gi.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"]); gi.set_defaults(func=global_info)
    go=sub.add_parser("global-online",help="run global-market paper observation, delayed reward, replay and asynchronous continual RL")
    go.add_argument("--data",required=True); go.add_argument("--state-dir",default=str(default_runtime_dir()/"global-online"))
    go.add_argument("--model-dir",default=os.environ.get("STOCKRL_MODEL_DIR",str(Path.home()/"Desktop"/"모델")),
                    help="directory containing only champion.pt and candidate.pt")
    go.add_argument("--window",type=int,default=128); go.add_argument("--horizon",type=str,default="1bar",
      help="outcome horizon: 30s, 1m, 5m, 1bar, 5bars. Durations resolve to the next observed bar at/after the target time.")
    go.add_argument("--fee",type=float,default=.001); go.add_argument("--slippage-bps",type=float,default=1.0)
    go.add_argument("--min-replay",type=int,default=8); go.add_argument("--batch-size",type=int,default=2)
    go.add_argument("--updates",type=int,default=2); go.add_argument("--lr",type=float,default=2e-6)
    go.add_argument("--replay-capacity",type=int,default=4096); go.add_argument("--max-observations",type=int,default=None)
    go.add_argument("--teacher-decisions",default=None,help="optional normalized CSV with date,action from public-teacher inference")
    go.add_argument("--teacher-replay",default=None,help="optional saved warm-start replay; its sampling share decays to zero after 10k self outcomes")
    go.add_argument("--teacher-symbol",default="MSFT")
    go.add_argument("--teacher-source",default=None,help="unique replay label for this teacher CSV")
    go.add_argument("--follow",action="store_true",help="watch an append-only CSV feed until Ctrl+C")
    go.add_argument("--poll-seconds",type=float,default=5.0)
    go.add_argument("--initial-lookback-bars",type=int,default=0,
                    help="on first boot only, paper-process this many latest existing bars before following new ones")
    go.add_argument("--candidate-every",type=int,default=512,
                    help="launch one asynchronous candidate update after this many additional replay experiences")
    go.add_argument("--initial-champion",default=None,
                    help="use this initial champion when absent, or migrate an incompatible legacy checkpoint safely")
    go.add_argument("--stride",type=int,default=1); go.add_argument("--wait-seconds",type=int,default=900)
    go.add_argument("--seed",type=int,default=7); go.add_argument("--device",default="auto",choices=["auto","cuda","mps","cpu"])
    go.set_defaults(func=global_online)
    lf=sub.add_parser("live-feed",help="poll public minute market data and append de-duplicated UTC bars for global-online")
    lf.add_argument("--config",default="configs/live_symbols.json"); lf.add_argument("--output",default=str(default_runtime_dir()/"live"/"market.csv"))
    lf.add_argument("--poll-seconds",type=float,default=15.0); lf.add_argument("--timeout",type=float,default=15.0)
    lf.add_argument("--once",action="store_true",help="poll every configured instrument once and exit")
    lf.add_argument("--stop-file",default=None,help="stop cleanly when this file is created")
    lf.add_argument("--provider-plugin",action="append",default=[],help="import a module that registers a MarketDataProviderAdapter; may be repeated")
    lf.add_argument("--max-cycles",type=int,default=None,help=argparse.SUPPRESS)
    def run_live_feed(args):
        import logging
        import importlib,os
        from .live_feed import LiveMarketCollector
        plugins=list(args.provider_plugin)+[x.strip() for x in os.environ.get("STOCKRL_MARKET_PROVIDER_PLUGINS","").split(",") if x.strip()]
        for plugin in plugins: importlib.import_module(plugin)
        logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
        LiveMarketCollector(args.config,args.output,args.poll_seconds,args.timeout,stop_file=args.stop_file).run(args.once,args.max_cycles)
    lf.set_defaults(func=run_live_feed)
    mf=sub.add_parser("mock-feed",help="stream real historical CSV bars at a controllable pace into the live paper feed")
    mf.add_argument("--source",default="data/global_market_daily.csv"); mf.add_argument("--output",default=str(default_runtime_dir()/"desktop"/"mock_market.csv"))
    mf.add_argument("--bars",type=int,default=24); mf.add_argument("--interval-seconds",type=float,default=.5)
    mf.add_argument("--start-offset",type=int,default=0); mf.add_argument("--max-cycles",type=int,default=None)
    mf.add_argument("--stop-file",default=None)
    def run_mock_feed(args):
        import logging
        from .mock_feed import replay_market_csv
        logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
        result=replay_market_csv(args.source,args.output,args.bars,args.interval_seconds,args.start_offset,args.max_cycles,args.stop_file)
        print(json.dumps(result,indent=2))
    mf.set_defaults(func=run_mock_feed)
    ui=sub.add_parser("desktop",help="open the Windows paper-trading and continual-learning dashboard")
    ui.add_argument("--runtime",default=str(default_runtime_dir()/"desktop")); ui.add_argument("--device",default="auto",choices=["auto","cuda","mps","cpu"])
    def run_desktop(args):
        from .desktop import run_desktop_app
        run_desktop_app(args.runtime,args.device)
    ui.set_defaults(func=run_desktop)
    web=sub.add_parser("web",help="open responsive browser dashboard and start the paper agent")
    web.add_argument("--host",default="127.0.0.1",help="bind address; localhost by default")
    web.add_argument("--port",type=int,default=8766); web.add_argument("--runtime",default=str(default_runtime_dir()))
    web.add_argument("--model-dir",default=None,help="directory for champion/candidate checkpoints")
    web.add_argument("--device",default="auto",choices=["auto","cuda","mps","cpu"])
    web.add_argument("--candidate-every",type=int,default=4096); web.add_argument("--fee",type=float,default=.001)
    web.add_argument("--horizon",default="1m",help="paper outcome horizon: 30s, 1m, or 5m")
    web.add_argument("--config",default="configs/live_symbols.json",help="market universe/provider config")
    web.add_argument("--initial-champion",default=None,help="seed checkpoint in the model directory")
    web.add_argument("--no-auto-start",action="store_true",help="open dashboard without starting feed/model")
    web.add_argument("--no-browser",action="store_true",help=argparse.SUPPRESS)
    def run_web(args):
        from .web_app import serve
        model_dir=args.model_dir or os.environ.get("STOCKRL_MODEL_DIR") or str(Path.home()/"Desktop"/"모델")
        serve(args.host,args.port,args.runtime,args.device,args.candidate_every,args.fee,
              not args.no_auto_start,not args.no_browser,args.horizon,args.config,args.initial_champion,model_dir)
    web.set_defaults(func=run_web)
    args=p.parse_args(); args.func(args)


if __name__=="__main__": main()

