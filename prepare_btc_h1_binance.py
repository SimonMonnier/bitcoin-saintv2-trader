# -*- coding: utf-8 -*-
"""Bougies H1 du BTC depuis 2017, construites sur Binance, pour le jeu H1.

2026-09-28, pistes 1 et 2 de la liste des pistes non explorees avec le jeu :
  1  BEAUCOUP PLUS D'HISTOIRE. Le jeu M15 n'apprenait que sur l'historique
     MT5 (moins de deux ans). Binance publie les bougies du BTC depuis aout
     2017 : neuf ans, plusieurs cycles haussiers, baissiers et de range. Le
     prix Vantage suit Binance a quelques points de base pres ; on garde donc
     le prix Binance et on applique les COUTS Vantage.
  2  UN HORIZON PLUS LONG. En H1 l'ATR median du BTC vaut plusieurs dizaines
     de points de base pour ~3.5 de cout : le cout pese bien moins par coup
     qu'en M15.

LES COUTS VANTAGE, MESURES ET NON SUPPOSES. Le spread de chaque bougie est
le spread median Vantage du meme creneau (jour de la semaine x heure UTC),
mesure sur les bougies M15 MT5 de 2023 a 2026 : il porte le pic du passage
de minuit serveur. Le swap acheteur (-20 %/an) est compte par le jeu, sur la
duree de chaque coup (`JeuConfig.swap_achat_bps_jour`).

TOUTES LES COLONNES LISENT LE PASSE. Fenetres glissantes arriere, moyennes
exponentielles, financement verse au plus tard a la CLOTURE de la bougie
(jointure arriere sur l'heure de cloture). Le __main__ le verifie en coupant
les donnees et en recalculant.

    python prepare_btc_h1_binance.py
"""
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SOURCE = "klines_5m_spot_BTCUSDT.pkl"
PORTAGE = "cache_portage.pkl"
M15_VANTAGE = "cache_m15_mt5.pkl"
SORTIE = "data_cache_BTCUSD_H1_BINANCE.pkl"
CHAUFFE = 800                 # bougies d'amorce (la plus longue fenetre : 720)

RETOURS = (1, 2, 4, 8, 24, 72, 168)
EMAS = (20, 50, 200, 800)

FEATURE_COLS_H1 = (
    [f"ret_{k}h" for k in RETOURS]
    + ["atr_rel_168", "rv_24_168", "rsi_14", "rsi_56", "rsi_336"]
    + [f"ema_{k}_dev" for k in EMAS]
    + ["pos_24", "pos_168", "pos_720", "creux_720", "rebond_720"]
    + ["flux_1", "flux_4", "flux_24", "vol_z", "trades_z", "creux_x_flux"]
    + ["heure_sin", "heure_cos", "jour_sin", "jour_cos", "week_end"]
    + ["funding_der", "funding_moy3", "funding_dispo"]
    + ["dist_rond"]
)


def h1_binance() -> pd.DataFrame:
    d = pd.read_pickle(SOURCE)
    d["time"] = pd.to_datetime(d["time"])
    d = d.drop_duplicates("time").set_index("time").sort_index()
    agg = d.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "quote_vol": "sum", "nb_trades": "sum",
         "taker_buy_base": "sum"})
    n5 = d["close"].resample("1h", label="left", closed="left").count()
    agg = agg[n5 >= 6].dropna(subset=["open", "close"])
    return agg.reset_index()


def profil_spread() -> pd.Series:
    """Spread median Vantage (bps) par (jour de semaine, heure UTC)."""
    m = pd.read_pickle(M15_VANTAGE)["barres"]["BTCUSD"][["time", "spread_bps"]].copy()
    m = m[m["spread_bps"] > 0]
    ny = (pd.to_datetime(m["time"]) - pd.Timedelta(hours=7)).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT")
    m = m[~ny.isna()]
    utc = ny[~ny.isna()].dt.tz_convert("UTC").dt.tz_localize(None)
    cle = utc.dt.weekday * 24 + utc.dt.hour
    return m["spread_bps"].groupby(cle.to_numpy()).median()


def _rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    g = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    p = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + g / p.replace(0, np.nan))).to_numpy()


def features(d: pd.DataFrame, funding: pd.Series, spread: pd.Series) -> pd.DataFrame:
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
        d[f"ret_{k}h"] = (lc - pd.Series(lc).shift(k).to_numpy()) / a_rel
    d["atr_rel_168"] = atr / pd.Series(atr).rolling(168, min_periods=168).mean().to_numpy()
    r1 = pd.Series(np.diff(lc, prepend=np.nan))
    d["rv_24_168"] = (r1.rolling(24, min_periods=24).std() / r1.rolling(168, min_periods=168).std()).to_numpy()
    d["rsi_14"], d["rsi_56"], d["rsi_336"] = _rsi(c, 14), _rsi(c, 56), _rsi(c, 336)
    for k in EMAS:
        e = cs.ewm(span=k, adjust=False).mean().to_numpy()
        d[f"ema_{k}_dev"] = (c - e) / atr
    for k in (24, 168, 720):
        hh = pd.Series(h).rolling(k, min_periods=k).max().to_numpy()
        ll = pd.Series(l).rolling(k, min_periods=k).min().to_numpy()
        d[f"pos_{k}"] = (c - ll) / np.where(hh - ll > 0, hh - ll, np.nan)
        if k == 720:
            d["creux_720"] = (c - hh) / atr
            d["rebond_720"] = (c - ll) / atr
    v = d["volume"].to_numpy(np.float64)
    ratio = pd.Series(d["taker_buy_base"].to_numpy(np.float64) / np.where(v > 0, v, np.nan))
    base = ratio.rolling(720, min_periods=240).mean()
    for k in (1, 4, 24):
        d[f"flux_{k}"] = (ratio.rolling(k, min_periods=k).mean() - base).to_numpy()
    lv = pd.Series(np.log1p(v))
    d["vol_z"] = ((lv - lv.rolling(720, min_periods=240).mean())
                  / lv.rolling(720, min_periods=240).std()).to_numpy()
    lt = pd.Series(np.log1p(d["nb_trades"].to_numpy(np.float64)))
    d["trades_z"] = ((lt - lt.rolling(720, min_periods=240).mean())
                     / lt.rolling(720, min_periods=240).std()).to_numpy()
    creux24 = (c - pd.Series(h).rolling(24, min_periods=24).max().to_numpy()) / atr
    d["creux_x_flux"] = creux24 * d["flux_4"].to_numpy()
    t = pd.to_datetime(d["time"])
    hr = t.dt.hour.to_numpy()
    wd = t.dt.weekday.to_numpy()
    d["heure_sin"], d["heure_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d["jour_sin"], d["jour_cos"] = np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)
    d["week_end"] = (wd >= 5).astype(float)
    # Le financement verse AU PLUS TARD a la cloture de la bougie (t + 1 h).
    fu = funding.sort_index()
    f = pd.DataFrame({"fin": t + pd.Timedelta(hours=1)})
    fdf = pd.DataFrame({"fin": fu.index, "f": fu.to_numpy(),
                        "f3": fu.rolling(3, min_periods=1).mean().to_numpy()})
    j = pd.merge_asof(f, fdf, on="fin", direction="backward")
    d["funding_dispo"] = j["f"].notna().astype(float).to_numpy()
    d["funding_der"] = (j["f"].fillna(0.0) * 1e4).to_numpy()
    d["funding_moy3"] = (j["f3"].fillna(0.0) * 1e4).to_numpy()
    u = 10.0 ** np.round(np.log10(0.01 * c))
    d["dist_rond"] = (c - np.round(c / u) * u) / atr
    cle = wd * 24 + hr
    d["spread_bar"] = spread.reindex(cle).fillna(spread.median()).to_numpy()
    return d


def construit():
    d = h1_binance()
    fu = pd.read_pickle(PORTAGE)["BTCUSDT"][["time", "funding"]].copy()
    fu["time"] = pd.to_datetime(fu["time"])
    fu = fu.drop_duplicates("time").set_index("time")["funding"]
    sp = profil_spread()
    f = features(d, fu, sp)
    return f, fu, sp, d


def verifie_causalite(d, fu, sp, essais=12, graine=5):
    plein = features(d, fu, sp)
    rng = np.random.default_rng(graine)
    pts = rng.integers(CHAUFFE + 100, len(d) - 1, essais)
    for t in pts:
        coupe = features(d.iloc[:t + 1].reset_index(drop=True), fu, sp)
        a = plein.loc[t, FEATURE_COLS_H1].to_numpy(np.float64)
        b = coupe.loc[t, FEATURE_COLS_H1].to_numpy(np.float64)
        if not np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9):
            bad = [FEATURE_COLS_H1[i] for i in np.flatnonzero(~np.isclose(a, b, equal_nan=True))]
            return f"FUITE a la bougie {t} : {bad}"
    return f"causalite verifiee sur {len(pts)} bougies : les colonnes ne lisent pas l'avenir"


def main() -> int:
    f, fu, sp, d = construit()
    print(f"H1 Binance : {len(d):,} bougies, {d['time'].iloc[0]} -> {d['time'].iloc[-1]} UTC")
    print(f"spread Vantage par creneau : median {sp.median():.2f} bps, "
          f"de {sp.min():.2f} a {sp.max():.2f} bps")
    print(verifie_causalite(d, fu, sp))
    f = f.iloc[CHAUFFE:].reset_index(drop=True)
    manque = f[FEATURE_COLS_H1 + ["atr_14"]].isna().mean()
    print(f"apres {CHAUFFE} bougies d'amorce : {len(f):,} bougies ; colonnes avec des trous : "
          + (", ".join(f"{k} {100 * v:.1f}%" for k, v in manque[manque > 0].items()) or "aucune"))
    f[FEATURE_COLS_H1] = f[FEATURE_COLS_H1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr_bps = (f["atr_14"] / f["close"] * 1e4)
    print(f"ATR H1 median {atr_bps.median():.1f} bps (p10 {atr_bps.quantile(0.1):.1f}) ; "
          f"{len(FEATURE_COLS_H1)} colonnes")
    f.to_pickle(SORTIE)
    print(f"ecrit {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
