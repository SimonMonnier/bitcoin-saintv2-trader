# -*- coding: utf-8 -*-
"""Bougies M5 du BTC depuis 2017, Binance et Coinbase, pour le jeu M5.

2026-09-28, demande du proprietaire : « on va passer sur du M5, puisqu'on a
des donnees M5 sur Coinbase et Binance que l'on n'a pas sur MT5 ; on va
trader le M5 par jour » (une partie = une journee UTC).

LE PRIX ET LES COUTS : ceux de `prepare_btc_h1_binance` — prix Binance
(Vantage le suit a quelques points de base pres), spread Vantage mesure par
creneau (jour x heure UTC), glissements et swap comptes par le jeu.

LES COLONNES sont celles du H1 ramenees a l'echelle de la bougie de 5
minutes (fenetres en bougies : 12 = une heure, 288 = un jour, 2016 = une
semaine), plus ce que seul le M5 peut lire :
  prime_cb       l'ecart de prix Coinbase / Binance, en bps, moins sa
                 moyenne du jour : la demande americaine au comptant contre
                 le marche mondial
  prime_cb_chg   sa variation sur 15 minutes
  prime_cb_dispo 1 si Coinbase a une bougie a cet instant

TOUT LIT LE PASSE : fenetres arriere, moyennes exponentielles, financement
et Coinbase joints a la CLOTURE de la bougie. Le __main__ le verifie.

    python prepare_btc_m5.py
"""
import os
import sys

import numpy as np
import pandas as pd

import prepare_btc_h1_binance as PH

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SOURCE = PH.SOURCE                        # klines_5m_spot_BTCUSDT.pkl
COINBASE = "cache_coinbase_5m_BTCUSD.pkl"
SORTIE = "data_cache_BTCUSD_M5_BINANCE.pkl"
CHAUFFE = 2100                            # une semaine d'amorce et un peu plus

RETOURS = (1, 3, 6, 12, 36, 72, 288, 864)          # 5 min ... 3 jours
EMAS = (20, 50, 200, 800)

FEATURE_COLS_M5 = (
    [f"ret_{k}" for k in RETOURS]
    + ["atr_rel_2016", "rv_288_2016", "rsi_14", "rsi_56", "rsi_168"]
    + [f"ema_{k}_dev" for k in EMAS]
    + ["pos_12", "pos_288", "pos_2016", "creux_2016", "rebond_2016"]
    + ["flux_1", "flux_3", "flux_12", "vol_z", "trades_z", "creux_x_flux"]
    + ["prime_cb", "prime_cb_chg", "prime_cb_dispo"]
    + ["heure_sin", "heure_cos", "jour_sin", "jour_cos", "week_end"]
    + ["funding_der", "funding_moy3", "funding_dispo"]
    + ["dist_rond"]
)


def binance_m5() -> pd.DataFrame:
    d = pd.read_pickle(SOURCE)
    d["time"] = pd.to_datetime(d["time"])
    return d.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def coinbase_m5():
    if not os.path.exists(COINBASE):
        return None
    c = pd.read_pickle(COINBASE)[["time", "close"]].rename(columns={"close": "cb_close"})
    c["time"] = pd.to_datetime(c["time"])
    return c.drop_duplicates("time")


def features(d: pd.DataFrame, cb, funding: pd.Series, spread: pd.Series) -> pd.DataFrame:
    d = d.copy()
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    cs = pd.Series(c)
    pc = cs.shift(1).to_numpy()
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    d["atr_14"] = atr
    a_rel = atr / c
    lc = np.log(c)
    for k in RETOURS:
        d[f"ret_{k}"] = (lc - pd.Series(lc).shift(k).to_numpy()) / a_rel
    d["atr_rel_2016"] = atr / pd.Series(atr).rolling(2016, min_periods=2016).mean().to_numpy()
    r1 = pd.Series(np.diff(lc, prepend=np.nan))
    d["rv_288_2016"] = (r1.rolling(288, min_periods=288).std()
                        / r1.rolling(2016, min_periods=2016).std()).to_numpy()
    d["rsi_14"], d["rsi_56"], d["rsi_168"] = PH._rsi(c, 14), PH._rsi(c, 56), PH._rsi(c, 168)
    for k in EMAS:
        e = cs.ewm(span=k, adjust=False).mean().to_numpy()
        d[f"ema_{k}_dev"] = (c - e) / atr
    for k in (12, 288, 2016):
        hh = pd.Series(h).rolling(k, min_periods=k).max().to_numpy()
        ll = pd.Series(l).rolling(k, min_periods=k).min().to_numpy()
        d[f"pos_{k}"] = (c - ll) / np.where(hh - ll > 0, hh - ll, np.nan)
        if k == 2016:
            d["creux_2016"] = (c - hh) / atr
            d["rebond_2016"] = (c - ll) / atr
    v = d["volume"].to_numpy(np.float64)
    ratio = pd.Series(d["taker_buy_base"].to_numpy(np.float64) / np.where(v > 0, v, np.nan))
    base = ratio.rolling(8640, min_periods=2016).mean()
    for k in (1, 3, 12):
        d[f"flux_{k}"] = (ratio.rolling(k, min_periods=k).mean() - base).to_numpy()
    lv = pd.Series(np.log1p(v))
    d["vol_z"] = ((lv - lv.rolling(8640, min_periods=2016).mean())
                  / lv.rolling(8640, min_periods=2016).std()).to_numpy()
    lt = pd.Series(np.log1p(d["nb_trades"].to_numpy(np.float64)))
    d["trades_z"] = ((lt - lt.rolling(8640, min_periods=2016).mean())
                     / lt.rolling(8640, min_periods=2016).std()).to_numpy()
    creux12 = (c - pd.Series(h).rolling(12, min_periods=12).max().to_numpy()) / atr
    d["creux_x_flux"] = creux12 * d["flux_3"].to_numpy()
    # COINBASE, a la meme bougie (les deux sont etiquetees par leur ouverture)
    if cb is not None:
        m = pd.DataFrame({"time": d["time"]}).merge(cb, on="time", how="left")
        p = (m["cb_close"].to_numpy(np.float64) / c - 1.0) * 1e4
        ps = pd.Series(p)
        d["prime_cb_dispo"] = np.isfinite(p).astype(float)
        d["prime_cb"] = (ps - ps.rolling(288, min_periods=48).mean()).to_numpy()
        d["prime_cb_chg"] = (ps - ps.shift(3)).to_numpy()
    else:
        d["prime_cb_dispo"], d["prime_cb"], d["prime_cb_chg"] = 0.0, np.nan, np.nan
    t = pd.to_datetime(d["time"])
    hr = t.dt.hour.to_numpy() + t.dt.minute.to_numpy() / 60.0
    wd = t.dt.weekday.to_numpy()
    d["heure_sin"], d["heure_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d["jour_sin"], d["jour_cos"] = np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)
    d["week_end"] = (wd >= 5).astype(float)
    fu = funding.sort_index()
    f = pd.DataFrame({"fin": t + pd.Timedelta(minutes=5)})
    fdf = pd.DataFrame({"fin": fu.index, "f": fu.to_numpy(),
                        "f3": fu.rolling(3, min_periods=1).mean().to_numpy()})
    j = pd.merge_asof(f, fdf, on="fin", direction="backward")
    d["funding_dispo"] = j["f"].notna().astype(float).to_numpy()
    d["funding_der"] = (j["f"].fillna(0.0) * 1e4).to_numpy()
    d["funding_moy3"] = (j["f3"].fillna(0.0) * 1e4).to_numpy()
    u = 10.0 ** np.round(np.log10(0.01 * c))
    d["dist_rond"] = (c - np.round(c / u) * u) / atr
    cle = wd * 24 + t.dt.hour.to_numpy()
    d["spread_bar"] = spread.reindex(cle).fillna(spread.median()).to_numpy()
    return d


def verifie_causalite(d, cb, fu, sp, essais=10, graine=5):
    plein = features(d, cb, fu, sp)
    rng = np.random.default_rng(graine)
    for t in rng.integers(CHAUFFE + 9000, len(d) - 1, essais):
        coupe = features(d.iloc[:t + 1].reset_index(drop=True), cb, fu, sp)
        a = plein.loc[t, FEATURE_COLS_M5].to_numpy(np.float64)
        b = coupe.loc[t, FEATURE_COLS_M5].to_numpy(np.float64)
        if not np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9):
            bad = [FEATURE_COLS_M5[i] for i in np.flatnonzero(~np.isclose(a, b, equal_nan=True))]
            return f"FUITE a la bougie {t} : {bad}"
    return f"causalite verifiee sur {essais} bougies : les colonnes ne lisent pas l'avenir"


def main() -> int:
    d = binance_m5()
    cb = coinbase_m5()
    fu = pd.read_pickle(PH.PORTAGE)["BTCUSDT"][["time", "funding"]].copy()
    fu["time"] = pd.to_datetime(fu["time"])
    fu = fu.drop_duplicates("time").set_index("time")["funding"]
    sp = PH.profil_spread()
    print(f"Binance M5 : {len(d):,} bougies, {d['time'].iloc[0]} -> {d['time'].iloc[-1]} UTC ; "
          f"Coinbase : " + ("absent" if cb is None else
                            f"{len(cb):,} bougies, {cb['time'].min()} -> {cb['time'].max()}"))
    print(verifie_causalite(d.iloc[-60_000:].reset_index(drop=True), cb, fu, sp))
    f = features(d, cb, fu, sp).iloc[CHAUFFE:].reset_index(drop=True)
    manque = f[FEATURE_COLS_M5 + ["atr_14"]].isna().mean()
    print(f"apres {CHAUFFE} bougies d'amorce : {len(f):,} bougies ; trous : "
          + (", ".join(f"{k} {100 * v:.1f}%" for k, v in manque[manque > 0].items()) or "aucun"))
    f[FEATURE_COLS_M5] = f[FEATURE_COLS_M5].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr_bps = f["atr_14"] / f["close"] * 1e4
    print(f"ATR M5 median {atr_bps.median():.1f} bps (p10 {atr_bps.quantile(0.1):.1f}) ; "
          f"Coinbase present sur {100 * f['prime_cb_dispo'].mean():.1f} % des bougies ; "
          f"{len(FEATURE_COLS_M5)} colonnes")
    f.to_pickle(SORTIE)
    print(f"ecrit {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
