"""Replay a FROZEN tree candidate on every minute with the actual environment."""
import argparse,json,pickle
from pathlib import Path
import training as t
import numpy as np
import pandas as pd
from economic_learning import causal_features


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='experiments/entry-research-sl4-20260914')
    ap.add_argument('--model',default='trees_expanded')
    args=ap.parse_args();root=Path(__file__).resolve().parent;folder=root/args.run
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    result=next(r for r in json.loads((folder/'results.json').read_text(encoding='utf-8')) if r['model']==args.model)
    if not result['calibration_accepted']:raise ValueError('Candidate rejected by calibration')
    cfg=t.PPOConfig(**meta['config']);cfg.use_vol_curriculum=False
    start,end=meta['bounds']['validation']
    frame=pd.read_pickle(root/'data_cache_BTCUSD_20221215.pkl')
    frame=frame.dropna(subset=t.FEATURE_COLS+['atr_14']).reset_index(drop=True).iloc[:end]
    if list(t.FEATURE_COLS)!=meta['feature_sets']['current']:
        raise ValueError('Feature schema changed since candidate training')
    columns=meta['feature_sets'][args.model.removeprefix('trees_')]
    with (folder/f'{args.model}.pkl').open('rb') as f:model=pickle.load(f)
    features=causal_features(frame,t.FEATURE_COLS,True)
    scores=model.predict(features[columns].iloc[start:end].to_numpy(np.float32))
    # Raw market features are irrelevant to this precomputed entry policy;
    # they remain present so the production environment API is exercised.
    data=t.MarketData(frame,t.FEATURE_COLS)
    rows=[]
    for multiplier in (1.,1.5):
        for seed in (71,72,73):
            cost_cfg=t.PPOConfig(**vars(cfg))
            for key in ('spread_bps','entry_slippage_bps','slippage_bps','fee_rate'):
                setattr(cost_cfg,key,getattr(cfg,key)*multiplier)
            np.random.seed(seed)
            env=t.BTCTradingEnvDiscrete(data,cost_cfg)
            env.idx=env.start_idx=start;env.end_idx=end
            done=False
            while not done:
                pred=scores[env.idx-start];side=int(np.argmax(pred))
                action=side if pred[side]>result['threshold'] and env.position==0 else 2
                _,_,done,_,info=env.step(action)
            pnl=np.array(env.trades_pnl)
            rows.append({'seed':seed,'cost_multiplier':multiplier,'trades':len(pnl),
                'net_pnl':float(pnl.sum()),'profit_factor':float(pnl[pnl>0].sum()/max(-pnl[pnl<0].sum(),1e-8)),
                'drawdown':env.max_dd,'done_reason':info['done_reason'],
                'bars_completed':env.idx-start,'bars_available':end-start})
            print(json.dumps(rows[-1]),flush=True)
    (folder/f'{args.model}_minute_replay.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
