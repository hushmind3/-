"""Verify downloaded native policies, append frozen modules, then reload one PT.

No learn(), optimizer step, original weight rewrite, or original expert probe.
"""
import argparse
import cloudpickle
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, PPO, SAC
from stockrl.moe_stock_policies import StockPolicyExpert, INDICATORS, make_policy
from stockrl.trading_moe import TradingMoE, EvidenceAdapter, parameter_digest
from stockrl.paths import EXPERT_ASSETS_DIR, EXPERT_WEIGHTS_DIR, TRADING_MOE_CHECKPOINT, expert_weight_path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def download(root):
    """Official repositories only; pin snapshots and retain original downloaded files."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    root.mkdir(parents=True,exist_ok=True)
    tasks=[];provenance={}
    repos={'dapo':'Ruijian-Zha/FinRL-DAPO-SR','msft':'maksimprivalov/RLTradingAgent','finrl':'tpeiqi/FinRL_trading_bot'}
    for tag,repo in repos.items():
        response=requests.get(f'https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1',timeout=30);response.raise_for_status()
        tree=response.json();revision=tree['sha'];provenance[tag]=dict(repo=repo,revision=revision)
        for entry in tree['tree']:
            name=entry['path']
            if entry['type']=='blob' and name.endswith(('.py','.zip','.csv','.md','.txt','.sh','.ipynb')) and not name.startswith(('logs/','log/','automation/','transaction_actions/')):
                tasks.append((root/tag/name,f'https://raw.githubusercontent.com/{repo}/{revision}/{name}'))
    for tag,repo,files in [('adilbai','Adilbai/stock-trading-rl-agent',
        ['final_model.zip','scaler.pkl','config.json','dataprocessor.py','enviromentcreator.py','README.md']),
        ('dapo','rz2689/finrl-dapo-grpo-sentiment-risk',['model_rl.pth'])]:
        response=requests.get('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=30);response.raise_for_status()
        model=response.json();revision=model['sha'];provenance[tag+'_weights']=dict(repo=repo,revision=revision)
        for name in files:tasks.append((root/tag/name,f'https://huggingface.co/{repo}/resolve/{revision}/{name}'))
    response=requests.get('https://huggingface.co/api/datasets/benstaf/nasdaq_2013_2023',timeout=30);response.raise_for_status()
    revision=response.json()['sha'];provenance['dapo_data']=dict(repo='benstaf/nasdaq_2013_2023',revision=revision)
    for name in ['risk','sentiment']:
        filename=f'trade_data_deepseek_{name}_2019_2023.csv'
        tasks.append((root/'dapo'/filename,f'https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/{revision}/{filename}'))
    def fetch(task):
        path,url=task;path=expert_weight_path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.download')
            with requests.get(url,stream=True,timeout=120) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024*1024):stream.write(block)
            temporary.replace(path)
        return dict(path=str(path.resolve()),url=url,bytes=path.stat().st_size,sha256=file_hash(path))
    with ThreadPoolExecutor(max_workers=6) as pool:files=list(pool.map(fetch,tasks))
    write_json(root/'downloads.json',dict(revisions=provenance,files=files))
    print('Official source/weight files ready:',len(files),flush=True)


def verify(root, device):
    dapo = pd.read_csv(root / 'dapo/trade_data_deepseek_risk_2019_2023.csv')
    sentiment = pd.read_csv(root / 'dapo/trade_data_deepseek_sentiment_2019_2023.csv')
    dapo['llm_sentiment'] = sentiment.llm_sentiment
    # Neutral score 3 is the author's own missing-news rule, not a synthetic market.
    dapo[['llm_sentiment', 'llm_risk']] = dapo[['llm_sentiment', 'llm_risk']].fillna(3.)
    dapo = dapo[dapo.date == '2023-01-03'].sort_values('tic')
    finrl = pd.read_csv(root / 'finrl/data/train_data.csv')
    msft = pd.read_csv(root / 'msft/data/msft_test.csv')
    scaler = pickle.loads((root / 'adilbai/scaler.pkl').read_bytes())
    scaler_spec = dict(features=scaler.feature_names_in_.tolist(), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
    common = dict(transaction_cost=.001)
    definitions = [
        ('stock_dapo', 'FinRL-DAPO-SR', 'dapo/model_rl.pth', None, 'dapo', sorted(dapo.tic.unique()), dapo),
        ('stock_adilbai', 'Adilbai Stock Trading RL Agent', 'adilbai/final_model.zip', PPO, 'adilbai',
            ['AAPL','AMZN','GOOGL','MSFT','TSLA'], finrl[finrl.tic.str.upper().isin(['AAPL','AMZN','GOOGL','MSFT','TSLA'])]),
        ('stock_msft_ppo', 'RLTradingAgent MSFT PPO', 'msft/ppo_trader.zip', PPO, 'msft', ['MSFT'], msft),
        ('stock_finrl_a2c', 'FinRL trading bot A2C', 'finrl/trained_models/agent_a2c.zip', A2C, 'a2c', sorted(finrl.tic.str.upper().unique()), finrl),
        ('stock_finrl_ppo', 'FinRL trading bot PPO', 'finrl/trained_models/agent_ppo.zip', PPO, 'ppo', sorted(finrl.tic.str.upper().unique()), finrl),
        ('stock_finrl_sac', 'FinRL trading bot SAC', 'finrl/trained_models/agent_sac.zip', SAC, 'sac', sorted(finrl.tic.str.upper().unique()), finrl),
    ]
    experts, inputs, report = {}, {}, []
    for key, name, relative, algorithm, kind, universe, frame in definitions:
        path = expert_weight_path(root / relative)
        spec = dict(common, kind=kind, universe=universe, schema=key+'_native_v1')
        if kind == 'dapo':
            weights = torch.load(path, map_location='cpu', weights_only=True)
            state = weights['model_state_dict']
            spec.update(observation_size=state['pi.mu_net.0.weight'].shape[1], hmax=100, indicators=INDICATORS,
                architecture_source=(root/'dapo/dapo_algorithm.py').read_text(encoding='utf-8'),
                activation='Tanh', training_epoch=weights['epoch'])
            policy = make_policy(spec); policy.load_state_dict(state, strict=True)
        else:
            loaded = algorithm.load(path, device='cpu')
            if loaded.num_timesteps <= 0: raise ValueError('no pretrained training provenance')
            policy = loaded.policy
            spec.update(observation_size=policy.observation_space.shape[0],
                constructor=cloudpickle.dumps(policy._get_constructor_parameters()), training_timesteps=int(loaded.num_timesteps))
            if kind in ('a2c','ppo','sac'): spec.update(hmax=20, indicators=INDICATORS)
            if kind == 'msft': spec['features']=['log_return','sma20','sma50','rsi14','macd','volume_change']
            if kind == 'adilbai':
                spec.update(initial_balance=10000, scaler=scaler_spec,
                    preprocessing_source=(root/'adilbai/dataprocessor.py').read_text(encoding='utf-8'),
                    native_config=json.loads((root/'adilbai/config.json').read_text(encoding='utf-8')))
        entry = dict(id=key, name=name, backend='stock_policy', role='stock_trading_policy', stock_policy=spec,
            parameters=sum(p.numel() for p in policy.parameters()), dtype='FP32', checkpoint_file=path.name,
            checkpoint_bytes=path.stat().st_size, checkpoint_sha256=file_hash(path))
        if 'Ticker' not in frame.columns and 'tic' not in frame.columns:
            frame=frame.copy();frame['Ticker']='MSFT'
        frame=frame.tail(17*400) if kind == 'adilbai' else frame
        date_col='date' if 'date' in frame.columns else 'Date'
        as_of=str(frame[date_col].max())
        account=dict(cash=1000000. if kind in ('dapo','a2c','ppo','sac') else 10000., positions={})
        account['nav']=account['cash']
        requested=universe if kind != 'adilbai' else ['MSFT']
        snapshot=dict(symbols=requested, as_of=as_of, stock_policy_history=frame.to_dict('records'), policy_account=account)
        expert=StockPolicyExpert(policy,entry)
        data=expert.prepare_input(snapshot)
        if data is None: raise ValueError(f'{key}: missing native historical input')
        before=parameter_digest(expert)
        packet=expert(None,data,device)
        if before != parameter_digest(expert): raise ValueError('native policy modified during inference')
        # Confirm reconstruction uses embedded config and actual strict state, not a fresh random policy.
        rebuilt=StockPolicyExpert.restore(entry,policy.state_dict())
        repeated=rebuilt(None,data,device)
        np.testing.assert_allclose(packet['native_output'],repeated['native_output'],atol=1e-6)
        experts[key]=expert;inputs[key]=data
        report.append(dict(expert=key,name=name,filename=path.name,file_bytes=path.stat().st_size,
            parameters=entry['parameters'],dtype='FP32',universe=universe,
            native_action=('84 signed share actions' if kind=='dapo' else '17 signed share actions' if kind in ('a2c','ppo','sac') else
                '[action_type 0=HOLD/1=BUY/2=SELL, fraction]' if kind=='adilbai' else '0=SELL/1=HOLD/2=BUY'),
            observation_shape=list(np.asarray(data['observations']).shape),native_output_shape=packet['raw_policy_shape'],
            raw_output=packet['raw_policy_output'],common_output=packet['common_output'],
            hash=before,training_timesteps=spec.get('training_timesteps'),training_epoch=spec.get('training_epoch'),
            inference_seconds=packet['worker_seconds']))
        print(key,entry['parameters'],'native inference OK',round(packet['worker_seconds'],4),flush=True)
    write_json(root/'native-verification.json',report)
    write_json(root/'native-inputs.json',inputs)
    torch.save({'entries':{k:e.entry for k,e in experts.items()},'states':{k:e.models[0].state_dict() for k,e in experts.items()}},expert_weight_path(root/'verified-policies.pt'))


def package(root, checkpoint):
    model, optimizer = TradingMoE.load_checkpoint(checkpoint)
    original = {k:parameter_digest(e) for k,e in model.experts.items()}
    original_controller = {k:v.detach().clone() for k,v in model.controller.state_dict().items()}
    verified=torch.load(expert_weight_path(root/'verified-policies.pt'),map_location='cpu',weights_only=True)
    new_ids=list(verified['entries'])
    if model.config.get('stock_policy_ids'):raise ValueError('stock policies are already registered; do not repack/reset learned adapters')
    model.config['stock_policy_ids']=new_ids
    for key,entry in verified['entries'].items():
        model.experts[key]=StockPolicyExpert.restore(entry,verified['states'][key])
        model.adapters[key]=EvidenceAdapter(8)
        model.config['feature_sizes'][key]=8
        model.config['native_module_counts'][key]=1
    from stockrl.trading_moe import VerticalController
    model.controller=VerticalController(model.config['feature_sizes'],new_ids)
    current=model.controller.state_dict()
    for k,v in original_controller.items():
        if k not in current or current[k].shape != v.shape:raise ValueError('existing controller structure changed')
        current[k]=v
    model.controller.load_state_dict(current,strict=True)
    model.metadata['stock_policy_integration']=dict(original_expert_hashes=original,
        source_urls=['https://github.com/Ruijian-Zha/FinRL-DAPO-SR',
          'https://huggingface.co/Adilbai/stock-trading-rl-agent','https://github.com/maksimprivalov/RLTradingAgent',
          'https://github.com/tpeiqi/FinRL_trading_bot'], no_training_performed=True)
    model.metadata['pre_expansion_optimizer_state']=optimizer
    # Old optimizer parameter indices cannot be silently applied to expanded groups.
    # Preserve the old state separately; training constructs current explicit groups.
    backup=checkpoint.with_name('TradingMoE.before-stock-policies.pt')
    if not backup.exists():
        import shutil
        shutil.copy2(checkpoint,backup)
    model.save_checkpoint(checkpoint)
    report=dict(expert_count=len(model.experts),total_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_bytes=checkpoint.stat().st_size,optimizer_updates_preserved=model.optimizer_updates,
        original_expert_hashes=original,original_controller_preserved=True,new_experts=new_ids)
    write_json(root/'package-report.json',report)
    publish(root,report,verified['entries'])
    print(json.dumps({k:v for k,v in report.items() if k!='original_expert_hashes'}),flush=True)


def publish(root,report=None,entries=None):
    if report is None:report=json.loads((root/'package-report.json').read_text(encoding='utf-8'))
    if entries is None:entries=torch.load(expert_weight_path(root/'verified-policies.pt'),map_location='cpu',weights_only=True)['entries']
    registry_path=Path('runtime/trading_moe/registry.json')
    if registry_path.exists():
        from stockrl.expert_registry import atomic_json
        registry=json.loads(registry_path.read_text(encoding='utf-8'))
        verified_report=json.loads((root/'native-verification.json').read_text(encoding='utf-8'))
        verified_inputs=json.loads((root/'native-inputs.json').read_text(encoding='utf-8'))
        for row in verified_report:
            key=row['expert'];entry=entries[key]
            folder='dapo' if key=='stock_dapo' else 'adilbai' if key=='stock_adilbai' else 'msft' if key=='stock_msft_ppo' else 'finrl/trained_models'
            relative=str(expert_weight_path(root/folder/row['filename']).resolve())
            output=root/(key+'-raw.json');write_json(output,dict(native_output=row['raw_output'],raw_policy_output=row['raw_output'],
                common_output=row['common_output'],output_shape=row['native_output_shape'],
                input_shapes={'observation':row['observation_shape']},as_of=verified_inputs[key]['as_of'],units='native_policy_action'))
            registry['experts']=[e for e in registry['experts'] if e['id']!=key]
            registry['experts'].append(dict(id=key,backend='stock_policy',name=entry['name'],role='주식 매매 정책',
                parameters=entry['parameters'],weight_bytes=entry['parameters']*4,dtype='FP32',checkpoint_bytes=entry['checkpoint_bytes'],
                files=[dict(path=relative,bytes=row['file_bytes'],sha256=entry['checkpoint_sha256'],archive=False)],
                universe=row['universe'],native_action=row['native_action'],verified=True,frozen=True,loaded=False,active=False,
                probe=dict(output_shape=row['native_output_shape'],input_shapes={'observation':row['observation_shape']},forward_seconds=row['inference_seconds']),
                router_selected=False,last_inference_seconds=row['inference_seconds'],last_output_shape=row['native_output_shape'],
                raw_output_path=str(output),raw_output_origin='independent_verification',error=None,worker=None,
                license='MIT' if key=='stock_adilbai' else 'Not declared in downloaded repository',last_peak_vram_bytes=0))
        atomic_json(registry_path,registry)
        status_path=Path('runtime/trading_moe/native_vertical_run/worker_status.json')
        if status_path.exists():
            status=json.loads(status_path.read_text(encoding='utf-8'));status.update(parameters=report['total_parameters'],expert_count=report['expert_count'])
            atomic_json(status_path,status)


def fresh(root,checkpoint,device):
    # Fail if any original model/scaler file is opened in this new process.
    forbidden={str(p.resolve()).casefold() for base in (root,root.parent/'checkpoints',EXPERT_WEIGHTS_DIR) for p in base.rglob('*') if p.suffix in ('.pth','.zip','.pkl','.pt','.safetensors')}
    def audit(event,args):
        if event=='open' and isinstance(args[0],str) and str(Path(args[0]).resolve()).casefold() in forbidden:
            raise RuntimeError('fresh load tried to read an external policy weight/scaler')
    sys.addaudithook(audit)
    started=time.perf_counter();model,optimizer_state=TradingMoE.load_checkpoint(checkpoint)
    loaded=time.perf_counter()
    original_moments=model.metadata.get('pre_expansion_optimizer_state')
    if original_moments:
        assert optimizer_state is not None and len(optimizer_state['state'])==len(original_moments['state'])
        optimizer=torch.optim.Adam(model.parameter_groups(),lr=.0002)
        optimizer.load_state_dict(optimizer_state)
    inputs=json.loads((root/'native-inputs.json').read_text(encoding='utf-8'))
    verification=json.loads((root/'native-verification.json').read_text(encoding='utf-8'))
    model.set_learning_device(device)
    results=[]
    for row in verification:
        key=row['expert'];e=model.experts[key]
        if parameter_digest(e)!=row['hash']:raise ValueError('embedded pretrained weight hash changed')
        output=e(model.root,inputs[key],device)
        np.testing.assert_allclose(output['common_output'][0]['target_weight'],row['common_output'][0]['target_weight'],atol=1e-6)
        np.testing.assert_allclose(output['raw_policy_output'],row['raw_output'],atol=1e-6)
        symbols=output['symbols']
        account=torch.zeros(1,len(symbols),16,device=device);account[...,0]=1.
        snapshot=dict(as_of=output['as_of'],symbols=symbols,currencies={s:'USD' for s in symbols},
            current_weights={s:0. for s in symbols},expert_inputs={})
        with torch.no_grad():decision,_=model(snapshot,account,packets=[output],device=device)
        assert all(torch.isfinite(t).all() for t in _.values())
        # No stock policy applies to Samsung or ETH even if a native source is available.
        assert not e.applicable('A005930') and not e.applicable('ETHUSDT')
        results.append(dict(expert=key,registered=True,frozen=all(not p.requires_grad for p in e.parameters()),
            inference_ok=True,hash_unchanged=True,common_output=output['common_output'],
            controller_actions=decision['trading_output']['actions'],seconds=output['worker_seconds']))
    baseline=model.metadata['stock_policy_integration']['original_expert_hashes']
    for key,digest in baseline.items():
        if parameter_digest(model.experts[key])!=digest:raise ValueError('original expert weight changed')
    for symbol in ['A005930','ETHUSDT']:
        account=torch.zeros(1,1,16,device=device)
        snapshot=dict(as_of='2026-10-03',symbols=[symbol],currencies={symbol:'USD'},current_weights={symbol:0.},expert_inputs={})
        with torch.no_grad():decision,out=model(snapshot,account,packets=[],device=device)
        assert not out['policy_validity'].any()
    # One same-date MSFT decision sees all six real stock opinions. Portfolio
    # policies still receive 84/17 native assets; adaptation exposes only MSFT.
    stamp='2023-01-03'
    finrl=pd.read_csv(root/'finrl/data/train_data.csv')
    msft=pd.read_csv(root/'msft/data/msft_test.csv');msft['Ticker']='MSFT'
    dapo=pd.read_csv(root/'dapo/trade_data_deepseek_risk_2019_2023.csv')
    sentiment=pd.read_csv(root/'dapo/trade_data_deepseek_sentiment_2019_2023.csv')
    dapo['llm_sentiment']=sentiment.llm_sentiment
    dapo[['llm_sentiment','llm_risk']]=dapo[['llm_sentiment','llm_risk']].fillna(3.)
    symbols=['MSFT','ETHUSDT','A005930']
    joint=dict(as_of=stamp,symbols=symbols,currencies={s:'USD' for s in symbols},current_weights={s:0. for s in symbols},
        policy_account=dict(cash=10000.,nav=10000.,positions={}),expert_inputs={})
    for key in model.controller.stock_policy_ids:
        source=dapo if key=='stock_dapo' else msft if key=='stock_msft_ppo' else finrl
        datum=model.experts[key].prepare_input({**joint,'stock_policy_history':source.to_dict('records')})
        if datum is None:raise ValueError(f'{key}: same-date MSFT native input missing')
        joint['expert_inputs'][key]=datum
    account=torch.zeros(1,3,16,device=device);account[...,0]=1.
    with torch.no_grad():decision,output=model(joint,account,device=device)
    assert len(decision['used_experts'])==6 and output['policy_validity'][0,0].sum()==6
    assert not output['policy_validity'][0,1:].any()
    assert all(torch.isfinite(value).all() for value in output.values())
    report=dict(load_seconds=loaded-started,expert_count=len(model.experts),new_experts=results,
        original_14_hashes_unchanged=True,symbol_masks_passed=True,external_weight_files_read=False,
        original_optimizer_moments_preserved=bool(original_moments),optimizer_updates_preserved=model.optimizer_updates,
        joint_msft=dict(as_of=stamp,used_experts=decision['used_experts'],actions=decision['trading_output']['actions'],
            target_weights=decision['trading_output']['target_weights'],policy_router_probabilities=output['policy_router_probabilities'].tolist(),
            seconds=decision['decision_seconds']),
        no_training_performed=True,parameters=sum(p.numel() for p in model.parameters()),checkpoint_bytes=checkpoint.stat().st_size)
    write_json(root/'fresh-process-report.json',report)
    print('FRESH PROCESS:',len(results),'new native policies OK;',len(baseline),'original hashes preserved;',report['parameters'],'parameters',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['download','verify','package','publish','fresh'])
    parser.add_argument('--root',type=Path,default=EXPERT_ASSETS_DIR/'stock-policies')
    parser.add_argument('--checkpoint',type=Path,default=TRADING_MOE_CHECKPOINT)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='download':download(args.root)
    elif args.stage=='verify':verify(args.root,args.device)
    elif args.stage=='package':package(args.root,args.checkpoint)
    elif args.stage=='publish':publish(args.root)
    else:fresh(args.root,args.checkpoint,args.device)
