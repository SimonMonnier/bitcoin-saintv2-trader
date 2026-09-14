"""Ablation sur validation : meme modele et memes fenetres, couts conserves.

Diagnostic exploratoire, pas un test hors echantillon final. La calibration
des seuils precede chronologiquement les episodes mesures.
"""
import argparse
import json
from pathlib import Path
import sys
import training as t
import numpy as np
import pandas as pd
import torch


def flat_probabilities(policy, data, indices, cfg, device):
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(indices), 128):
            batch = indices[offset:offset + 128]
            features = np.stack([data.features[i-cfg.lookback:i] for i in batch])
            extra = np.zeros((len(batch), cfg.lookback, 4), np.float32)
            extra[:, :, 3] = 1
            obs = torch.as_tensor(np.concatenate([features, extra], axis=-1), device=device)
            logits, _ = policy(obs)
            outputs.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(outputs)


def simulate(data, cfg, starts, probabilities, thresholds, seed):
    env = t.BTCTradingEnvDiscrete(data, cfg)
    rows = []
    for episode, start in enumerate(starts):
        # Memes dates et graines pour chaque scenario. Les tirages de couts
        # divergent ensuite si les actions divergent : pas de faux appariement.
        np.random.seed(seed + episode)
        env.reset()
        env.idx = env.start_idx = int(start)
        env.end_idx = int(start) + cfg.episode_length
        done = False
        while not done:
            action = 2
            if env.position == 0:
                pb, ps = [float(x) for x in probabilities[episode][env.idx-start, :2]]
                okb, oks = pb >= thresholds[0], ps >= thresholds[1]
                if okb and oks:
                    action = 0 if pb-thresholds[0] >= ps-thresholds[1] else 1
                elif okb:
                    action = 0
                elif oks:
                    action = 1
            _, _, done, _, info = env.step(action)
        # Liquidation terminale explicite au marche, avec frais et spread.
        terminal = 0.0
        if env.position:
            price = float(data.close[env.idx-1])
            price *= 1 - env.position * cfg.slippage_bps / 20000
            price = env._apply_micro(price, -env.position, is_entry=False)
            terminal = (env.position * (price-env.entry_price) * env.current_size
                        - cfg.fee_rate * price * env.current_size)
        pnl = list(env.trades_pnl)
        if env.position:
            pnl.append(terminal)
        final_equity = env.capital + terminal
        final_dd = max(env.max_dd, (env.peak_capital-final_equity)/max(env.peak_capital, 1e-8))
        rows.append({'episode': episode, 'seed': seed, 'start': int(start),
                     'net_pnl': final_equity - cfg.initial_capital,
                     'trades': len(pnl), 'gross_profit': sum(x for x in pnl if x > 0),
                     'gross_loss': -sum(x for x in pnl if x <= 0),
                     'wins': sum(x > 0 for x in pnl), 'drawdown': final_dd,
                     'terminal_liquidation': bool(env.position)})
    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='experiments/profitability_20260914_v2')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / args.run
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    cfg = t.PPOConfig(lookback=manifest['config']['lookback'],
                      episode_length=3000, use_vol_curriculum=False)
    df = pd.read_pickle(root / 'data_cache_BTCUSD_20221215.pkl')
    # Cache a parfois un emballage de metadonnees.
    if isinstance(df, dict):
        df = df['df']
    df = df.dropna(subset=t.FEATURE_COLS + ['atr_14']).reset_index(drop=True)
    if len(df) != manifest['total_rows'] or list(manifest['features']) != list(t.FEATURE_COLS):
        raise ValueError('Le cache ou les features ont change depuis cet entrainement.')
    a, b = manifest['train_rows'], manifest['val_rows']
    raw = df.iloc[a:a+b].reset_index(drop=True)
    norm = np.load(out / 'norm_stats_ohlc_indics.npz')
    stats = {'mean': norm['mean'], 'std': norm['std']}
    data = t.MarketData(raw, t.FEATURE_COLS, stats)
    device = t.get_device(cfg)
    policy = t.SAINTPolicySingleHead(n_features=t.OBS_N_FEATURES,
                                    d_model=cfg.d_model, num_blocks=2,
                                    heads=4, dropout=.05, ff_mult=2,
                                    max_len=cfg.lookback, n_actions=t.N_ACTIONS).to(device)
    checkpoint = out / 'bestprofit_pilot_saintv2_both_wf1.pth'
    policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()
    # Premier tiers pour calibrer; huit episodes disjoints dans les deux tiers suivants.
    calibration_indices = np.arange(cfg.lookback, b//3, 20)
    calibration = flat_probabilities(policy, data, calibration_indices, cfg, device)
    thresholds = [t.selective_threshold(calibration[:, side], .025) for side in (0, 1)]
    starts = np.linspace(b//3 + cfg.lookback, b-cfg.episode_length-2, 8, dtype=int)
    probs = [flat_probabilities(policy, data, np.arange(s, s+cfg.episode_length), cfg, device)
             for s in starts]
    rows = []
    scenarios = [('sans_filtre_sl2', [0., 0.], 2.),
                 ('top5_sl2', thresholds, 2.), ('top5_sl5', thresholds, 5.)]
    for name, thresholds_used, sl in scenarios:
        cfg.atr_sl_mult, cfg.atr_tp_mult = sl, sl * 1.4
        for seed in (10000, 20000, 30000):
            result = simulate(data, cfg, starts, probs, thresholds_used, seed)
            rows.extend(dict(scenario=name, **r) for r in result)
        print('SCENARIO COMPLETE', name, flush=True)
    details = pd.DataFrame(rows)
    details.to_csv(out / 'comparison_episodes.csv', index=False)
    summary = []
    for name, group in details.groupby('scenario', sort=False):
        summary.append({'scenario': name, 'mean_pnl_per_episode': group.net_pnl.mean(),
                        'profit_factor': group.gross_profit.sum()/max(group.gross_loss.sum(), 1e-8),
                        'win_rate': group.wins.sum()/max(group.trades.sum(), 1),
                        'trades_total_3_repetitions': int(group.trades.sum()),
                        'max_drawdown': group.drawdown.max(),
                        'positive_episodes': int((group.net_pnl > 0).sum()),
                        'episode_repetitions': len(group)})
    report = {'checkpoint': checkpoint.name, 'scenarios': summary,
              'calibration_thresholds': thresholds, 'test_evaluated': False,
              'note': 'Validation exploratoire deja utilisee pour selectionner le modele; '
                      '3 repetitions de couts sur 8 fenetres, pas 24 observations independantes.'}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
