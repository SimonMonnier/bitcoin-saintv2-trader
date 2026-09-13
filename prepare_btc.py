"""Prepare les donnees BTCUSD : fusion, cache disque et stats de normalisation.

A lancer AVANT toute mesure (SL/TP, apport des features) : ces mesures ont
besoin du dataframe fusionne et des stats de normalisation, alors que
training.py enchaine directement sur PPO une fois le cache construit.

    python prepare_btc.py
"""
import sys

from training import PPOConfig, load_mt5_data, \
    compute_and_save_global_norm_stats, load_global_norm_stats
from saint_core import FEATURE_COLS


def main() -> int:
    cfg = PPOConfig()
    print(f"symbole {cfg.symbol} | depuis {cfg.date_from:%Y-%m-%d} | "
          f"{len(FEATURE_COLS)} features")
    df = load_mt5_data(cfg)
    print(f"\n{len(df):,} bougies fusionnees "
          f"({df['time'].iloc[0]} -> {df['time'].iloc[-1]})")

    stats = load_global_norm_stats()
    if stats is None:
        stats = compute_and_save_global_norm_stats(df, FEATURE_COLS)
    print("\nstats de normalisation :")
    for c, m, s in zip(FEATURE_COLS, stats["mean"], stats["std"]):
        print(f"  {c:<18} moy {m:>+12.6f}   ecart-type {s:>12.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
