# -*- coding: utf-8 -*-
"""Le jeu M5 a plusieurs cryptos, avec EXACTEMENT les features du BTC.

2026-09-28, demande du proprietaire : « toutes les cryptos presentes sur MT5
avec Vantage et aussi sur Binance et Coinbase, avec exactement les memes
features que pour le BTC ; un seul modele, un petit cerveau par marche ».

QUELLES CRYPTOS. 37 symboles Vantage ont une paire USDT au comptant et un
perpetuel chez Binance ET une paire USD chez Coinbase (verification du
2026-09-28). Mais le spread Vantage, lu dans MT5 sur les 99 000 dernieres
bougies M5 (`telecharge_crypto_mt5.py`), rapporte a l'ATR M5 median :

    BTC 2.2 bps / 14 = 0.16    ZEC 13.7 / 45 = 0.30    ETH 10.7 / 19 = 0.55
    puis SKY 1.4, HBAR 2.4, SOL 2.7, XRP 2.8, BNB 2.8 ... WLD 26 fois l'ATR

Au-dela d'une fois l'ATR, le spread seul coute plus que le mouvement moyen
d'une bougie, et meme au 10e centile (SOL 37 bps) : aucun signal ne le
couvre. Sont gardes ceux dont le spread median reste sous l'ATR M5 : BTC,
ETH, ZEC.

LES FEATURES sont celles de `prepare_btc_m5.features`, a l'identique, chacune
lue sur SES sources : bougies 5 min Binance au comptant (flux acheteur,
nombre de trades), bougies 5 min Coinbase (ecart Coinbase / Binance),
financement du perpetuel Binance, spread Vantage par creneau (jour x heure
UTC). Plus une colonne `m_<marche>` par marche, qui dit au modele quel petit
cerveau utiliser.

    python prepare_multi_m5.py
"""
import sys

import numpy as np
import pandas as pd

import prepare_btc_h1_binance as PH
import prepare_btc_m5 as P5

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SORTIE = "data_cache_MULTI_M5.pkl"
MT5 = "cache_crypto_m5_mt5.pkl"
FUNDING = "cache_funding_crypto.pkl"
# symbole Vantage -> actif
MARCHES = {"BTCUSD": "BTC", "ETHUSD": "ETH", "ZECUSD": "ZEC"}
FEATURE_COLS_MULTI_M5 = list(P5.FEATURE_COLS_M5) + [f"m_{m}" for m in MARCHES]


def profil_spread_m5(barres: pd.DataFrame) -> pd.Series:
    """Spread median Vantage (bps) par (jour de semaine, heure UTC), comme
    `prepare_btc_h1_binance.profil_spread`, lu sur les bougies M5 MT5."""
    m = barres.loc[barres["spread"] > 0, ["time", "spread_bps"]].copy()
    ny = (pd.to_datetime(m["time"]) - pd.Timedelta(hours=7)).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT")
    m = m[~ny.isna()]
    utc = ny[~ny.isna()].dt.tz_convert("UTC").dt.tz_localize(None)
    cle = utc.dt.weekday * 24 + utc.dt.hour
    return m["spread_bps"].groupby(cle.to_numpy()).median()


def swap_bps_jour(pct_an: float) -> float:
    """Swap Vantage des cryptos (mode 5, % par an, negatif = paye) en bps
    du prix par jour de detention, compte comme un cout (positif)."""
    return max(-float(pct_an), 0.0) / 365.0 * 100.0


def sources(sym: str, mt5: dict, fu_cache: dict):
    a = MARCHES[sym]
    d = pd.read_pickle(f"klines_5m_spot_{a}USDT.pkl")
    d["time"] = pd.to_datetime(d["time"])
    d = d.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    try:
        cb = pd.read_pickle(f"cache_coinbase_5m_{a}USD.pkl")[["time", "close"]].rename(
            columns={"close": "cb_close"})
        cb["time"] = pd.to_datetime(cb["time"])
        cb = cb.drop_duplicates("time")
    except FileNotFoundError:
        cb = None
    if a == "BTC":
        fu = pd.read_pickle(PH.PORTAGE)["BTCUSDT"][["time", "funding"]].copy()
        fu["time"] = pd.to_datetime(fu["time"])
        fu = fu.drop_duplicates("time").set_index("time")["funding"]
        sp = PH.profil_spread()
    else:
        fu = fu_cache[f"{a}USDT"]
        sp = profil_spread_m5(mt5["barres"][sym])
    return d, cb, fu, sp


def main() -> int:
    mt5 = pd.read_pickle(MT5)
    fu_cache = pd.read_pickle(FUNDING)
    parts = []
    for sym in MARCHES:
        d, cb, fu, sp = sources(sym, mt5, fu_cache)
        if sym != "BTCUSD":
            print(f"{sym}: " + P5.verifie_causalite(d.iloc[-60_000:].reset_index(drop=True),
                                                     cb, fu, sp), flush=True)
        f = P5.features(d, cb, fu, sp).iloc[P5.CHAUFFE:].reset_index(drop=True)
        f[P5.FEATURE_COLS_M5] = (f[P5.FEATURE_COLS_M5].replace([np.inf, -np.inf], np.nan)
                                 .fillna(0.0))
        f["marche"] = sym
        for m in MARCHES:
            f[f"m_{m}"] = float(m == sym)
        inf = mt5["infos"][sym]
        f["swap_achat_bps_jour"] = swap_bps_jour(inf["swap_long"])
        f["swap_vente_bps_jour"] = swap_bps_jour(inf["swap_short"])
        atr_bps = f["atr_14"] / f["close"] * 1e4
        print(f"{sym}: {len(f):,} bougies {f['time'].iloc[0]} -> {f['time'].iloc[-1]}  "
              f"ATR M5 median {atr_bps.median():.1f} bps  spread Vantage median "
              f"{f['spread_bar'].median():.2f} bps  Coinbase {100 * f['prime_cb_dispo'].mean():.0f} %  "
              f"financement {100 * f['funding_dispo'].mean():.0f} %  swap "
              f"{f['swap_achat_bps_jour'].iloc[0]:.2f} / {f['swap_vente_bps_jour'].iloc[0]:.2f} bps/jour",
              flush=True)
        parts.append(f)
    # LE BTC DOIT ETRE CELUI DU JEU M5 SEUL, a l'identique.
    ref = pd.read_pickle(P5.SORTIE)
    b = parts[0]
    ok = (len(ref) == len(b) and (ref["time"].values == b["time"].values).all()
          and np.allclose(ref[P5.FEATURE_COLS_M5].to_numpy(np.float64),
                          b[P5.FEATURE_COLS_M5].to_numpy(np.float64), equal_nan=True))
    print(f"BTCUSD identique au cache du BTC seul ({P5.SORTIE}) : {'oui' if ok else 'NON'}")
    cols = (["time", "marche", "open", "high", "low", "close", "atr_14", "spread_bar",
             "swap_achat_bps_jour", "swap_vente_bps_jour"] + FEATURE_COLS_MULTI_M5)
    out = pd.concat([p[list(dict.fromkeys(cols))] for p in parts], ignore_index=True)
    out.to_pickle(SORTIE)
    print(f"ecrit {SORTIE} : {len(out):,} bougies, {len(FEATURE_COLS_MULTI_M5)} colonnes du jeu")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
