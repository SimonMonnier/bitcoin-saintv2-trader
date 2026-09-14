"""Reproducible CPU ablations with purged chronological splits and net costs.

The last 30% is not read for training/evaluation. This is an exploratory
entry-model comparison, not a deployment or a live trading process.
"""
import argparse
import json
import hashlib
import time
from pathlib import Path
import training as t
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from economic_learning import causal_features,net_trade_targets,simulate_entries
from research_models import TemporalPatchRegressor,NetReturnMLP


class SaintRegressor(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.policy=t.SAINTPolicySingleHead(features,d_model=80,max_len=25)
    def forward(self,x):
        return self.policy(x)[0][:,:2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='experiments/entry-research-20260914')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--train-samples',type=int,default=8000)
    parser.add_argument('--sl-mult',type=float,default=None)
    parser.add_argument('--fast',action='store_true',help='Trees and MLP ensemble only')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent;out=root/args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=t.PPOConfig();cfg.tick_noise_bps=0;cfg.use_be_trail=False
    if args.sl_mult is not None:
        cfg.atr_sl_mult=args.sl_mult;cfg.atr_tp_mult=args.sl_mult*1.4
    import shutil
    for name in ('training.py','saint_core.py','execution_quotes.py','economic_learning.py','research_models.py',Path(__file__).name):
        shutil.copy2(root/name,out/name)
    df=pd.read_pickle(root/'data_cache_BTCUSD_20221215.pkl')
    df=df.dropna(subset=t.FEATURE_COLS+['atr_14']).reset_index(drop=True)
    n=len(df);h=cfg.max_holding_bars
    # Keep the final 30% untouched. Development is inside the training era.
    bounds={'train':(400,int(.45*n)), 'dev':(int(.45*n),int(.55*n)),
            'calibration':(int(.55*n),int(.60*n)), 'validation':(int(.60*n),int(.70*n))}
    df=df.iloc[:int(.70*n)].copy()
    full=causal_features(df,t.FEATURE_COLS,True)
    compact=['close_ema_dev','returns','range_norm','mom_5','vol_rank',
             'close_h1_dev','rsi_14_h1','returns_h1','taker_ratio','ls_ratio_top']
    sets={'compact':compact,'current':list(t.FEATURE_COLS),'expanded':list(full.columns)}
    sample_limits={'train':args.train_samples,'dev':1600,'calibration':3000,'validation':6000}
    indices={};ys={};ends={}
    for name,(a,b) in bounds.items():
        # Labels end before the boundary, windows are strictly historical.
        indices[name]=np.unique(np.linspace(a+96,b-h-1,sample_limits[name]).astype(int))
        ys[name],ends[name]=net_trade_targets(df,indices[name],cfg)
    stress_y,stress_end=net_trade_targets(df,indices['validation'],cfg,1.5)
    manifest={'config':vars(cfg),'bounds':bounds,'indices':{k:v.tolist() for k,v in indices.items()},
              'total_rows':n,'reserved_from':int(.70*n),'feature_sets':sets,'args':vars(args),
              'costs':'fixed configured spread; half max adverse slippage; exit fee; stress x1.5',
              'evaluation':'irregular sampled entry grid, one position at a time, R units, no leverage compounding',
              'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                         ('training.py','saint_core.py','economic_learning.py','research_models.py',Path(__file__).name)}}
    (out/'manifest.json').write_text(json.dumps(manifest,default=str,indent=2),encoding='utf-8')
    results=[]
    def assess(name,preds,extra):
        # A positive predicted net edge is required. Threshold candidates are
        # frozen using calibration only; validation is never optimized here.
        cal=preds['calibration'];thresholds=[0.,.05,.1,.2,.4]
        options=[]
        for threshold in thresholds:
            score=simulate_entries(indices['calibration'],cal,ys['calibration'],ends['calibration'],threshold)
            options.append((score['net_R'] if score['trades']>=30 else -float('inf'),threshold))
        best,threshold=max(options)
        accepted=best>0
        # Also report the raw positive-edge strategy when calibration rejects
        # the model, so zero trades can never be sold as predictive success.
        raw=simulate_entries(indices['validation'],preds['validation'],ys['validation'],ends['validation'],0.)
        val=simulate_entries(indices['validation'],preds['validation'],ys['validation'],ends['validation'],threshold) if accepted else {'trades':0,'net_R':0.,'mean_R':0.,'profit_factor':0.,'win_rate':0.,'entries':[]}
        stress=simulate_entries(indices['validation'],preds['validation'],stress_y,stress_end,threshold) if accepted else dict(val)
        row={'model':name,'calibration_accepted':accepted,'threshold':threshold,
             'raw_positive_edge_validation':raw,'validation':val,'stress':stress,**extra}
        results.append(row)
        (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='raw_positive_edge_validation'}),flush=True)

    for feature_set in ('compact','current','expanded'):
        x=full[sets[feature_set]].to_numpy(np.float32)
        model=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,
               min_samples_leaf=80,l2_regularization=10,early_stopping=False,random_state=42))
        start=time.time();model.fit(x[indices['train']],ys['train'])
        preds={s:model.predict(x[indices[s]]) for s in ('calibration','validation')}
        import pickle
        with (out/f'trees_{feature_set}.pkl').open('wb') as f:pickle.dump(model,f)
        assess('trees_'+feature_set,preds,{'seconds':time.time()-start})

    # Neural ablations use identical input columns and labels. Only the
    # temporal encoder/window changes. Independent MLP seeds form an ensemble.
    columns=sets['current'];x=full[columns].to_numpy(np.float32)
    # Scaler fits TRAIN only; imputation is frozen at its means.
    tr=x[:bounds['train'][1]]
    mean=np.nanmean(tr,axis=0);std=np.nanstd(tr,axis=0).clip(1e-6)
    x=np.clip(np.nan_to_num((x-mean)/std),-5,5).astype(np.float32)
    np.savez(out/'neural_norm.npz',mean=mean,std=std,features=np.array(columns))
    ensemble=[]
    for name,lookback,seed in [('mlp_42',1,42),('mlp_43',1,43),('mlp_44',1,44),('saint_25',25,42),('patch_96',96,42)]:
        if args.fast and not name.startswith('mlp'):
            continue
        torch.manual_seed(seed);rng=np.random.default_rng(seed);start=time.time()
        model=(NetReturnMLP(len(columns)) if name.startswith('mlp') else
               SaintRegressor(len(columns)) if name.startswith('saint') else TemporalPatchRegressor(len(columns)))
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        best=float('inf');state=None
        def batch(ix):
            # full row i is lagged once, so last row i is the latest closed bar.
            rows=ix[:,None]-np.arange(lookback-1,-1,-1)[None,:]
            return torch.from_numpy(x[rows])
        def predict(which):
            model.eval();parts=[]
            with torch.no_grad():
                for i in range(0,len(indices[which]),128):
                    parts.append(model(batch(indices[which][i:i+128])).numpy())
            return np.concatenate(parts)
        for epoch in range(args.epochs):
            model.train();order=rng.permutation(len(indices['train']))
            for a in range(0,len(order),128):
                k=order[a:a+128];opt.zero_grad(set_to_none=True)
                prediction=model(batch(indices['train'][k]))
                # MSE estimates conditional expected NET R (unlike TP BCE).
                loss=nn.functional.mse_loss(prediction,torch.from_numpy(ys['train'][k].astype(np.float32)))
                if not torch.isfinite(loss):raise ValueError('Non-finite supervised loss')
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Non-finite gradient')
                opt.step()
            dev=float(np.mean((predict('dev')-ys['dev'])**2))
            if dev<best:
                best=dev;state={k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f'{name} epoch {epoch+1}/{args.epochs} dev MSE={dev:.5f}',flush=True)
        model.load_state_dict(state)
        torch.save({'state_dict':state,'purpose':'net_R_regression_NOT_PPO','features':columns,
                    'lookback':lookback,'seed':seed},out/f'{name}.pt')
        preds={s:predict(s) for s in ('calibration','validation')}
        if name.startswith('mlp'):ensemble.append(preds)
        assess(name,preds,{'seconds':time.time()-start,'dev_mse':best,'parameters':sum(p.numel() for p in model.parameters())})
    averaged={s:np.mean([p[s] for p in ensemble],axis=0) for s in ('calibration','validation')}
    assess('mlp_ensemble_3',averaged,{'members':3})
    print('DONE; no deployment; final 30% untouched',flush=True)


if __name__=='__main__':main()
