"""Pilot reproductible, sorties isolees et test final preserve."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import shutil

import training as t  # Initialise le runtime OpenMP avant numpy/torch.

import numpy as np
import pandas as pd
import torch


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--output', default='experiments/profitability_20260914')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--episode-length', type=int, default=3000)
    parser.add_argument('--val-episodes', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--sl-mult', type=float, default=2.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = (root / args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = t.PPOConfig(epochs=args.epochs, episodes_per_epoch=args.episodes,
                      val_episodes=args.val_episodes, episode_length=args.episode_length,
                      critic_warmup_epochs=args.warmup,
                      atr_sl_mult=args.sl_mult, atr_tp_mult=args.sl_mult * 1.4,
                      validation_selectivity=0.05, evaluate_test=False,
                      model_prefix='pilot_saintv2', side='both',
                      use_vol_curriculum=False)
    for name in ('training.py', 'saint_core.py', Path(__file__).name):
        shutil.copy2(root / name, out / name)
    frame = t.load_mt5_data(cfg)
    n = len(frame)
    a, b, c = int(n * .55), int(n * .15), int(n * .10)
    # La fin de l'historique ne participe ni aux stats ni a la selection.
    os.chdir(out)
    stats = t.compute_and_save_global_norm_stats(frame.iloc[:a], t.FEATURE_COLS)
    train, val, test = t.create_datasets_from_slices(
        frame, t.FEATURE_COLS, start=0, train_len=a, val_len=b,
        test_len=c, stats=stats)
    manifest = {'config': vars(cfg), 'seed': t.SEED,
                'features': t.FEATURE_COLS, 'train_rows': a, 'val_rows': b,
                'test_rows_reserved': c, 'total_rows': n,
                'normalization': 'train only', 'test_evaluated': False,
                'validation_note': 'Exploratoire; calibration sur les episodes de validation.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, default=str, indent=2), encoding='utf-8')
    # Reinitialiser apres la preparation pour une relance deterministe.
    random.seed(t.SEED)
    np.random.seed(t.SEED)
    torch.manual_seed(t.SEED)
    t.run_training_on_split(train, val, test, stats, cfg, suffix='_wf1')
    log = pd.read_csv(out / 'training_log_both_wf1.csv')
    log['val_pnl_per_episode'] = log.val_pnl / cfg.val_episodes
    summary = {'epochs_completed': len(log),
               'last_epoch': log.iloc[-1].to_dict(),
               'best_validation_pnl_epoch': log.loc[log.val_pnl.idxmax()].to_dict(),
               'profitable_validation_epochs': int((log.val_pnl > 0).sum()),
               'test_evaluated': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('EXPERIMENT COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
