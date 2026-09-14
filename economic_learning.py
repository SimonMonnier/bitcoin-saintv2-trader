"""Causal features and net-R targets for offline entry research.

Targets are NOT action probabilities or PPO checkpoints. Costs are explicit;
unresolved trades are liquidated rather than removed from the training set.
"""
import numpy as np
import pandas as pd


def downside_score(returns):
    """Non-annualized Sortino, zero target, RMS over all observations."""
    r = np.asarray(returns, dtype=np.float64)
    if not r.size:
        return 0.0
    if not np.isfinite(r).all():
        raise ValueError('Non-finite returns')
    downside = np.sqrt(np.mean(np.minimum(r, 0.) ** 2))
    return float(r.mean() / max(downside, 1e-8))


def causal_features(frame, base_columns, expanded=False):
    """Row i contains only observations available BEFORE open[i]."""
    x = frame[base_columns].astype(float).copy()
    if expanded:
        close = frame.close.astype(float)
        atr = frame.atr_14.astype(float).clip(lower=1e-8)
        for window in (5, 30, 120, 360):
            x[f'move_atr_{window}'] = (close - close.shift(window)) / atr
            x[f'efficiency_{window}'] = ((close - close.shift(window)) /
                close.diff().abs().rolling(window).sum().clip(lower=1e-8))
        x['atr_bps'] = atr / close * 10000
        x['body_atr'] = (close-frame.open)/atr
        x['upper_wick_atr'] = (frame.high-np.maximum(frame.open,close))/atr
        x['lower_wick_atr'] = (np.minimum(frame.open,close)-frame.low)/atr
    return x.shift(1).replace([np.inf, -np.inf], np.nan)


def net_trade_targets(frame, indices, cfg, cost_multiplier=1.):
    """Vectorized deterministic execution scenario matching training barriers.

    Uses configured spread and half the max adverse entry/exit slippage.
    Net R is per unit stop risk, fees at exit as in the current environment.
    No stochastic wick extensions, trailing, or account-level DD termination.
    """
    idx = np.asarray(indices, dtype=np.int64)
    horizon = cfg.max_holding_bars
    if horizon <= 0 or np.any(idx < 1) or np.any(idx+horizon >= len(frame)):
        raise ValueError('Every label must fit completely inside its split')
    op, hi, lo, cl, atr = [frame[c].to_numpy(float) for c in ('open','high','low','close','atr_14')]
    spread = cfg.spread_bps * cost_multiplier / 10000
    ent_slip = cfg.entry_slippage_bps * cost_multiplier / 20000
    out_slip = cfg.slippage_bps * cost_multiplier / 20000
    from saint_core import ATR_PLANCHER_FRAC
    outcomes=[]; exits=[]
    for side in (1,-1):
        entry = op[idx] * (1+spread if side==1 else 1) * (1+side*ent_slip)
        a = np.maximum(atr[idx-1], np.maximum(ATR_PLANCHER_FRAC*entry, 1e-8))
        risk = cfg.atr_sl_mult*a
        stop = entry-side*risk
        target = entry+side*cfg.atr_tp_mult*cfg.tp_shrink*a
        alive=np.ones(len(idx),bool); fill=np.zeros(len(idx)); end=idx+horizon
        quote_factor=1 if side==1 else 1+spread
        for h in range(horizon+1):
            j=idx+h
            stop_hit=(lo[j]*quote_factor<=stop) if side==1 else (hi[j]*quote_factor>=stop)
            tp_hit=(hi[j]*quote_factor>=target) if side==1 else (lo[j]*quote_factor<=target)
            hit=alive & (stop_hit|tp_hit|(h==horizon))
            gap=np.minimum(stop,op[j]*quote_factor) if side==1 else np.maximum(stop,op[j]*quote_factor)
            price=np.where(stop_hit,gap,np.where(tp_hit,target,cl[j]*quote_factor))
            price*=np.where(stop_hit|~tp_hit,1-side*out_slip,1.)
            fill[hit]=price[hit];end[hit]=j[hit];alive[hit]=False
        net=side*(fill-entry)-cfg.fee_rate*cost_multiplier*fill
        outcomes.append(net/risk);exits.append(end)
    return np.stack(outcomes,1),np.stack(exits,1)


def simulate_entries(indices, scores, outcomes, exits, threshold=0.):
    """One position at a time; no execution on the same bar as an exit."""
    available=-1; trades=[]; entries=[]
    for k,i in enumerate(indices):
        side=int(np.argmax(scores[k]))
        if i<=available or not np.isfinite(scores[k,side]) or scores[k,side]<=threshold:
            continue
        trades.append(float(outcomes[k,side]));entries.append(int(i))
        available=int(exits[k,side])
    r=np.asarray(trades)
    return {'trades':len(r),'net_R':float(r.sum()),'mean_R':float(r.mean()) if len(r) else 0.,
            'profit_factor':float(r[r>0].sum()/max(-r[r<0].sum(),1e-8)),
            'win_rate':float((r>0).mean()) if len(r) else 0.,'entries':entries}
