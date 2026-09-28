# -*- coding: utf-8 -*-
"""Le cache du jeu multi-marches H1 : le BTC (Binance) et six indices (MT5).

2026-09-28, demande du proprietaire : revenir a la configuration du run
kairos_jeu_h1_01 (+447 $ au test du fold 1, une position a la fois, 10
jetons par semaine) et AJOUTER des indices volatils a faible spread pour
augmenter le nombre de trades et le profit. Choix des indices : voir
`telecharge_h1_mt5.py`.

CE QUI EST COMMUN, CE QUI NE L'EST PAS
  Le modele SAINT est commun : il lit les colonnes que tous les marches ont
  (prix, volatilite, RSI, moyennes, position dans le range, volume relatif,
  horloge, nombres ronds) et une colonne par marche. Les colonnes propres au
  BTC — flux acheteur Binance, nombre de transactions, financement des
  perpetuels — ne vont qu'a SON expert LightGBM (`EXTRAS_BTC_H1`).

TOUT EST EN UTC. Binance l'est ; MT5 est a l'heure du serveur (New York
+ 7 h, ete comme hiver) : on retire 7 h, on situe l'heure a New York, et on
convertit. Les deux marches parlent alors du meme instant.

LES COUTS DE CHAQUE MARCHE
  spread   le spread median Vantage de chaque creneau (jour x heure UTC),
           mesure sur les 12 derniers mois de bougies H1 a spread non nul
           (l'historique ancien en porte beaucoup de nuls) ; pour le BTC,
           celui de `prepare_btc_h1_binance`
  swap     le swap du courtier, en bps du prix par jour, lu dans MT5 (le
           jeu le compte au prorata de la duree de chaque coup)

LES FENETRES DU WALK-FORWARD sont celles du run h1_01 : tous les marches sont
coupes a la periode du BTC (sept. 2017 -> sept. 2026), pour que les folds
tombent aux memes dates.

    python prepare_multi_h1.py
"""
import sys

import numpy as np
import pandas as pd

import prepare_btc_h1_binance as PH

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BTC_H1 = PH.SORTIE
MT5_H1 = "cache_h1_mt5.pkl"
SORTIE = "data_cache_MULTI_H1.pkl"
MARCHES = ["BTCUSD", "NAS100", "GER40", "UK100", "FRA40", "HK50", "US2000"]
SWAP_BTC = (20.0 / 365 * 100, 0.0)          # cout en bps/jour : achat, vente

COMMUNES = (
    [f"ret_{k}h" for k in PH.RETOURS]
    + ["atr_rel_168", "rv_24_168", "rsi_14", "rsi_56", "rsi_336"]
    + [f"ema_{k}_dev" for k in PH.EMAS]
    + ["pos_24", "pos_168", "pos_720", "creux_720", "rebond_720", "vol_z"]
    + ["heure_sin", "heure_cos", "jour_sin", "jour_cos", "week_end", "dist_rond"]
)
FEATURE_COLS_MULTI_H1 = COMMUNES + [f"m_{m}" for m in MARCHES]
EXTRAS_BTC_H1 = ["flux_1", "flux_4", "flux_24", "trades_z", "creux_x_flux",
                 "funding_der", "funding_moy3", "funding_dispo"]


def serveur_vers_utc(t: pd.Series) -> pd.Series:
    """Heure du serveur MT5 (New York + 7 h) -> UTC, heure d'ete comprise."""
    ny = (pd.to_datetime(t) - pd.Timedelta(hours=7)).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT")
    return ny.dt.tz_convert("UTC").dt.tz_localize(None)


def profil_spread_mt5(d: pd.DataFrame, point: float) -> pd.Series:
    """Spread median (bps) par (jour, heure UTC), sur les 12 derniers mois."""
    r = d[(d["spread"] > 0) & (d["time"] >= d["time"].max() - pd.Timedelta(days=365))]
    bps = r["spread"] * point / r["close"] * 1e4
    cle = r["time"].dt.weekday * 24 + r["time"].dt.hour
    return bps.groupby(cle.to_numpy()).median()


def swap_bps_jour(info: dict, prix: float):
    """(cout achat, cout vente) en bps du prix par jour ; negatif = credit.
    Les indices de Vantage sont factures en argent par lot et par jour, dans
    la devise du prix (modes 2 et 3)."""
    notion = float(info["contrat"]) * prix
    return (-float(info["swap_long"]) / notion * 1e4, -float(info["swap_short"]) / notion * 1e4)


def bloc_indice(nom, d, info, debut, fin):
    d = d.copy()
    d["time"] = serveur_vers_utc(d["time"])
    d = d.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    sp = profil_spread_mt5(d, float(info["point"]))
    base = pd.DataFrame({"time": d["time"], "open": d["open"], "high": d["high"], "low": d["low"],
                         "close": d["close"], "volume": d["tick_volume"].astype(float),
                         "quote_vol": 0.0, "nb_trades": d["tick_volume"].astype(float),
                         "taker_buy_base": d["tick_volume"].astype(float) / 2})
    f = PH.features(base, pd.Series(dtype=float, index=pd.DatetimeIndex([])), sp)
    f = f.iloc[PH.CHAUFFE:]
    f = f[(f["time"] >= debut) & (f["time"] <= fin)].reset_index(drop=True)
    sa, sv = swap_bps_jour(info, float(d["close"].iloc[-1]))
    return f, sp, sa, sv


def construit():
    btc = pd.read_pickle(BTC_H1)
    debut, fin = btc["time"].min(), btc["time"].max()
    mt = pd.read_pickle(MT5_H1)
    blocs, resume = [], []
    b = btc.copy()
    b["marche"] = "BTCUSD"
    b["swap_achat_bps_jour"], b["swap_vente_bps_jour"] = SWAP_BTC
    blocs.append(b)
    resume.append(("BTCUSD", len(b), b["time"].iloc[0], b["time"].iloc[-1],
                   float(b["spread_bar"].median()), *SWAP_BTC))
    for nom in MARCHES[1:]:
        f, sp, sa, sv = bloc_indice(nom, mt["barres"][nom], mt["infos"][nom], debut, fin)
        f["marche"] = nom
        f["swap_achat_bps_jour"], f["swap_vente_bps_jour"] = sa, sv
        for c in EXTRAS_BTC_H1:
            f[c] = np.nan
        blocs.append(f)
        resume.append((nom, len(f), f["time"].iloc[0], f["time"].iloc[-1],
                       float(f["spread_bar"].median()), sa, sv))
    d = pd.concat(blocs, ignore_index=True)
    for m in MARCHES:
        d[f"m_{m}"] = (d["marche"] == m).astype(float)
    d[COMMUNES] = d[COMMUNES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    garde = list(dict.fromkeys(["time", "marche", "open", "high", "low", "close", "atr_14",
                                "spread_bar", "swap_achat_bps_jour", "swap_vente_bps_jour"]
                               + FEATURE_COLS_MULTI_H1 + EXTRAS_BTC_H1))
    return d[garde], resume


def main() -> int:
    d, resume = construit()
    print(f"{'marche':8s} {'bougies':>8s}  {'debut':19s}  {'fin':19s}  spread   swap achat/vente (%/an)")
    for nom, n, a, b, sp, sa, sv in resume:
        print(f"{nom:8s} {n:8,d}  {str(a):19s}  {str(b):19s}  {sp:5.2f}   "
              f"{-sa * 365 / 100:+6.1f} / {-sv * 365 / 100:+6.1f}")
    atr_bps = d["atr_14"] / d["close"] * 1e4
    print("ATR H1 median par marche (bps) : "
          + "  ".join(f"{m} {atr_bps[d['marche'] == m].median():.1f}" for m in MARCHES))
    # Verification : le meme instant en UTC. L'ouverture de New York (9h30)
    # doit tomber a 13h30 UTC l'ete, 14h30 l'hiver.
    t = pd.Series(pd.to_datetime(["2024-07-08 16:30", "2024-01-08 16:30"]))
    u = serveur_vers_utc(t)
    print(f"controle UTC : 16h30 serveur -> {u.iloc[0]:%H:%M} UTC en juillet, "
          f"{u.iloc[1]:%H:%M} UTC en janvier (attendu 13:30 et 14:30)")
    d.to_pickle(SORTIE)
    print(f"ecrit {SORTIE} : {len(d):,} bougies, {len(FEATURE_COLS_MULTI_H1)} colonnes communes, "
          f"{len(EXTRAS_BTC_H1)} colonnes de l'expert du BTC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
