# -*- coding: utf-8 -*-
"""Le cache M15 multi-marches : un seul jeu pour le BTC, l'ETH, l'or et les indices.

2026-09-27, demande du proprietaire : « un seul jeu qui trade l'ETH, l'or et
les indices a petit spread et bonne volatilite ; peu importe le marche, je
veux maximiser le nombre de trades et le profit en gardant un drawdown bas ».

LES SOURCES : les bougies M15 du courtier (`telecharge_m15_mt5.py`), spread
de chaque bougie compris. Binance et Coinbase ne concernent que la crypto :
le jeu commun s'en passe, pour tous les marches.

LES FEATURES sont celles du BTC M15, sans les sources externes : prix M15,
contextes H1 et H4, tendance longue, horloge des annonces US, cout et
regime rapportes a leur creneau horaire, reperes de prix (le nombre rond
suit le prix du marche), rang du creux — plus une colonne par marche, pour
que le modele sache ce qu'il joue.

LES MARCHES SONT EMPILES PAR BLOC, chacun dans l'ordre du temps : le jeu
ne laisse aucun coup traverser d'un bloc a l'autre.

    python prepare_multi_m15.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import features_scalping as FS
import prepare_m5
from saint_core import FEATURE_COLS_H1, FEATURE_COLS_LIQ_TEMPS, FEATURE_COLS_M1

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SOURCE = "cache_m15_mt5.pkl"
SORTIE = "data_cache_MULTI_M15.pkl"
MARCHES = ["BTCUSD", "ETHUSD", "XAUUSD", "NAS100", "SP500", "DJ30", "GER40"]
MPB, JOUR = 15, 96
ECHELLES = [("1h", "_h1", 24), ("4h", "_h4", 6)]

_REDONDANTES = ("rsi_ok", "high_vol_regime")
COLONNES_MARCHE = [f"m_{m}" for m in MARCHES]
FEATURE_COLS_MULTI = (
    [c for c in FEATURE_COLS_M1 if c not in _REDONDANTES]
    + [c.replace("_h1", sfx) for sfx in ("_h1", "_h4") for c in FEATURE_COLS_H1
       if c.replace("_h1", "") not in _REDONDANTES]
    + FEATURE_COLS_LIQ_TEMPS + FS.COLONNES_HORLOGE + FS.COLONNES_REGIME
    + FS.COLONNES_ANCRES + ["creux_rang"] + COLONNES_MARCHE)
assert len(FEATURE_COLS_MULTI) == len(set(FEATURE_COLS_MULTI))


def cadre(b: pd.DataFrame) -> pd.DataFrame:
    """Les bougies MT5 d'un marche, au format de `prepare_m5.construit`.

    UN SPREAD A ZERO N'EST PAS UN SPREAD NUL : c'est une bougie ou MT5 ne
    l'a pas enregistre (35 % des bougies du DAX). Il est remplace par la
    mediane des bougies qui en ont un — sinon le jeu jouerait gratuitement.
    """
    sp = b["spread_bps"].astype(np.float64)
    med = float(sp[sp > 0].median()) if (sp > 0).any() else 1.0
    sp = sp.where(sp > 0, med)
    return pd.DataFrame({
        "time": b["time"], "open": b["open"].astype(np.float64),
        "high": b["high"].astype(np.float64), "low": b["low"].astype(np.float64),
        "close": b["close"].astype(np.float64),
        "volume": b["tick_volume"].astype(np.float64),
        "spread_bar": sp, "tick_n": b["tick_volume"].astype(np.float64),
        "tick_spread_moy": sp, "tick_spread_max": sp,
    }).sort_values("time").reset_index(drop=True)


def features_marche(b: pd.DataFrame, marche: str) -> pd.DataFrame:
    f = cadre(b)
    d, _c, _i = prepare_m5.construit(f, avec_flux=False, jour=JOUR, echelles=ECHELLES,
                                     avec_structures=False)
    d = FS.ajoute(d, mpb=MPB, sources=False,
                  unite_rond=FS.unite_ronde(float(f["close"].median())))
    for m in MARCHES:
        d[f"m_{m}"] = 1.0 if m == marche else 0.0
    d["marche"] = marche
    return d


def main() -> int:
    z = pd.read_pickle(SOURCE)
    blocs = []
    for m in MARCHES:
        if m not in z["barres"]:
            print(f"{m:8s} absent du cache", flush=True)
            continue
        d = features_marche(z["barres"][m], m)
        avant = len(d)
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=FEATURE_COLS_MULTI + ["atr_14"]).reset_index(drop=True)
        a = d["atr_14"] / d["close"] * 1e4
        print(f"{m:8s} {avant:6d} -> {len(d):6d} bougies  {d['time'].iloc[0]:%Y-%m-%d} -> "
              f"{d['time'].iloc[-1]:%Y-%m-%d}  ATR med {a.median():5.1f} bps  spread med "
              f"{d['spread_bar'].median():5.2f} bps  nombre rond "
              f"{FS.unite_ronde(float(d['close'].median())):g}", flush=True)
        blocs.append(d)
    out = pd.concat(blocs, ignore_index=True)
    for c in FEATURE_COLS_MULTI:
        out[c] = out[c].astype(np.float32)
    out.to_pickle(SORTIE)
    print(f"\n{SORTIE} : {len(out):,} bougies, {len(blocs)} marches, "
          f"{len(FEATURE_COLS_MULTI)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
