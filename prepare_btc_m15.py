# -*- coding: utf-8 -*-
"""Le cache M15 du BTC : les memes sources que le M1, en bougies de 15 minutes.

2026-09-27, demande du proprietaire : « recommence le jeu avec des bougies de
15 minutes, et pas M1 pour entrer ». Le modele voit des bougies M15 et ne
decide qu'a leur cloture ; ses contextes superieurs deviennent le H1 et le
H4.

POURQUOI CE PEUT CHANGER QUELQUE CHOSE. Le cout d'un trade est fixe (~3.3
bps) ; le mouvement d'une bougie grandit avec sa duree. L'ATR M15 vaut
environ quatre fois l'ATR M1 : le meme cout pese quatre fois moins par
unite de risque.

LES SOURCES SONT CELLES DU M1, AGREGEES — le brut du courtier (`BRUT`),
Binance et Coinbase deja alignes a l'heure du serveur par
`prepare_btc_m1.joint_sources`. Une bougie M15 porte l'heure de son
ouverture, comme MT5 :

    prix          premiere ouverture, plus haut, plus bas, derniere cloture
    spread        moyenne de la bougie ; ticks : somme, moyenne, maximum
    Binance       volumes et nombres de trades sommes, dernier prix
    Coinbase      dernier prix

LES FEATURES sont celles du M1 (`features_scalping`), leurs fenetres
converties de minutes en bougies par `mpb = 15` : la meme duree, pas le
meme nombre de lignes. `test_causalite_btc_m15.py` verifie qu'aucune ne lit
le futur.

    python prepare_btc_m15.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import features_scalping as FS
import prepare_btc_m1 as P1
import prepare_m5
from saint_core import (FEATURE_COLS_H1, FEATURE_COLS_LIQ_TEMPS, FEATURE_COLS_M1,
                        FEATURE_COLS_RANGS)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SORTIE = "data_cache_BTCUSD_M15.pkl"
MPB = 15
JOUR_M15 = 96
# LES CONTEXTES DU M15 : le H1 et le H4. (regle, suffixe, bougies par jour.)
ECHELLES_BTC_M15 = [("1h", "_h1", 24), ("4h", "_h4", 6)]

_REDONDANTES = ("rsi_ok", "high_vol_regime")
FEATURE_COLS_M15 = (
    [c for c in FEATURE_COLS_M1 if c not in _REDONDANTES]
    + [c.replace("_h1", sfx) for sfx in ("_h1", "_h4") for c in FEATURE_COLS_H1
       if c.replace("_h1", "") not in _REDONDANTES]
    + FEATURE_COLS_LIQ_TEMPS + FS.COLONNES_HORLOGE + FS.COLONNES_REGIME
    + FS.colonnes_flux(MPB) + FS.COLONNES_COINBASE + FS.COLONNES_ANCRES
    + FEATURE_COLS_RANGS + ["creux_x_flux"])
assert len(FEATURE_COLS_M15) == len(set(FEATURE_COLS_M15))


def agrege_m15(m1: pd.DataFrame) -> pd.DataFrame:
    """Des minutes (brut + Binance + Coinbase) aux bougies de 15 minutes."""
    g = m1.set_index("time").resample("15min", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1),
        "spread_bar": g["spread_bar"].mean(),
        "tick_n": g["tick_n"].sum(min_count=1),
        "tick_spread_moy": g["tick_spread_moy"].mean(),
        "tick_spread_max": g["tick_spread_max"].max(),
        "bn_close": g["bn_close"].last(),
        "bn_volume": g["bn_volume"].sum(min_count=1),
        "bn_quote_vol": g["bn_quote_vol"].sum(min_count=1),
        "nb_trades": g["nb_trades"].sum(min_count=1),
        "taker_buy_base": g["taker_buy_base"].sum(min_count=1),
        "cb_close": g["cb_close"].last(),
    })
    return out.dropna(subset=["open", "close"]).reset_index()


def features_m15(m15: pd.DataFrame) -> pd.DataFrame:
    d, _c, _i = prepare_m5.construit(m15, avec_flux=False, jour=JOUR_M15,
                                     echelles=ECHELLES_BTC_M15,
                                     avec_structures=False)
    return FS.ajoute(d, mpb=MPB)


def main() -> int:
    m1 = pd.read_pickle(P1.BRUT).sort_values("time").reset_index(drop=True)
    m1 = P1.joint_sources(m1)
    m15 = agrege_m15(m1)
    print(f"{len(m1):,} minutes -> {len(m15):,} bougies M15  "
          f"{m15['time'].iloc[0]} -> {m15['time'].iloc[-1]}", flush=True)
    d = features_m15(m15)
    absentes = [c for c in FEATURE_COLS_M15 if c not in d.columns]
    if absentes:
        raise RuntimeError(f"colonnes non produites : {absentes}")
    avant = len(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURE_COLS_M15 + ["atr_14"]).reset_index(drop=True)
    for c in FEATURE_COLS_M15:
        d[c] = d[c].astype(np.float32)
    d.to_pickle(SORTIE)
    a = d["atr_14"] / d["close"] * 1e4
    print(f"  chauffe : {avant:,} -> {len(d):,} bougies ; {len(FEATURE_COLS_M15)} features",
          flush=True)
    print(f"  ATR M15 : mediane {a.median():.1f} bps (p10 {a.quantile(0.1):.1f}, "
          f"p90 {a.quantile(0.9):.1f}) ; spread median {d['spread_bar'].median():.2f} bps",
          flush=True)
    print(f"{SORTIE} ecrit : {d['time'].iloc[0]} -> {d['time'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
