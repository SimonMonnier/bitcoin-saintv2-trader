# -*- coding: utf-8 -*-
"""Les sources BTC gratuites portent-elles de la DIRECTION ? — 2026-09-26.

Trois sources que le projet n'avait jamais lues (voir
`telecharge_sources_btc.py`) : les perpetuels Binance (flux agressif et
ecart au spot), leurs metriques de positions, et Coinbase. Chaque candidate
est mesuree contre le rendement BRUT futur — d'ouverture bid a ouverture
bid, ce que la ligne `avantage brut` de la validation appelle le brut.

LA FENETRE DE TEST N'EST JAMAIS LUE. Le walk-forward du run
`saintv2_btc_m1_scalp01` teste a partir de la ligne 655 224 (fold 1), puis
sur les deux tranches suivantes : toute ligne a partir de la est du test
pour un fold ou un autre. La mesure s'arrete donc AVANT, rendements futurs
compris :

    apprentissage   [0 : 514 819)        le train du fold 1
    mesure          [514 819 : 655 224)  calibration + validation

TROIS MESURES, de la plus brute a la plus proche du modele :

  1. CORRELATION DE RANG avec le rendement a 1, 5, 15 et 60 minutes, sur
     des rendements DISJOINTS (un point toutes les h minutes), dans les deux
     fenetres. Un signal reel a le MEME signe dans les deux.
  2. L'ECART ENTRE DECILES EXTREMES, en bps : ce que rapporterait, brut,
     d'acheter le decile haut et de vendre le decile bas. C'est le chiffre
     a comparer au cout — 2.7 bps par trade en validation.
  3. L'APPORT A UN MODELE : un LightGBM appris sur les 68 colonnes du run,
     puis sur les 68 + les nouvelles, et ce qu'il capte en brut sur la
     fenetre de mesure.

Une centaine de tests : a ce nombre, |t| > 3.5 est le seuil ou le hasard
cesse d'etre une explication suffisante.

    python mesure_sources_btc.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import features_scalping as FS
from saint_core import FEATURE_COLS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIN_APPRENTISSAGE = 514_819
FIN_MESURE = 655_224          # debut du test du fold 1 : jamais au-dela
HORIZONS = (1, 5, 15, 60)
COUT_BPS = 2.7


def _serveur(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["time"] = FS.serveur_depuis_utc(pd.to_datetime(d["time"]))
    return d.drop_duplicates("time").sort_values("time")


def charge() -> pd.DataFrame:
    cols = ["time", "open", "close", "bn_close", "bn_volume", "taker_buy_base"]
    d = pd.read_pickle("data_cache_BTCUSD_M1.pkl")
    base = d[list(dict.fromkeys(["time", "open", "close"] + FEATURE_COLS))].copy()
    del d
    # Le spot Binance n'est pas garde dans le cache : on le relit.
    bn = FS.aligne_binance(pd.read_pickle("cache_binance_1m_BTCUSDT.pkl"))
    base = base.merge(bn[["time", "bn_close", "bn_volume", "taker_buy_base"]],
                      on="time", how="left")

    pp = _serveur(pd.read_pickle("cache_perp_1m_BTCUSDT.pkl"))
    pp = pp.rename(columns={"close": "pp_close", "volume": "pp_vol",
                            "taker_buy_volume": "pp_tb"})[
        ["time", "pp_close", "pp_vol", "pp_tb"]]
    pr = _serveur(pd.read_pickle("cache_premium_1m_BTCUSDT.pkl"))
    pr = pr.rename(columns={"close": "prem"})[["time", "prem"]]
    cb = _serveur(pd.read_pickle("cache_coinbase_1m_BTCUSD.pkl"))
    cb = cb.rename(columns={"close": "cb_close", "volume": "cb_vol"})[
        ["time", "cb_close", "cb_vol"]]
    mt = pd.read_pickle("cache_metrics_5m_BTCUSDT.pkl")
    # LES METRIQUES SONT DES INSTANTANES DE 5 MINUTES. Leur heure de
    # publication n'est pas documentee : on ne les rend disponibles que
    # CINQ minutes apres leur horodatage, par prudence.
    mt["time"] = pd.to_datetime(mt["time"]) + pd.Timedelta(minutes=5)
    mt = _serveur(mt)

    for x in (pp, pr, cb):
        base = base.merge(x, on="time", how="left")
    base = pd.merge_asof(base.sort_values("time"), mt, on="time",
                         direction="backward")
    return base.reset_index(drop=True)


def verifie_alignement(d: pd.DataFrame) -> None:
    a = d["close"].pct_change()
    for nom, c in (("perpetuel", "pp_close"), ("coinbase", "cb_close")):
        b = d[c].pct_change()
        ok = a.notna() & b.notna()
        j = pd.DataFrame({"a": a[ok], "b": b[ok], "m": d["time"][ok].dt.to_period("M")})
        r = j.groupby("m").apply(lambda x: np.corrcoef(x["a"], x["b"])[0, 1])
        print(f"  alignement {nom:10s} : correlation par mois min {r.min():+.3f} "
              f"mediane {r.median():+.3f}  couverture {100*d[c].notna().mean():.1f} %")


def candidates(d: pd.DataFrame) -> list:
    n = []
    # --- PERPETUELS : flux agressif ---
    s = 2.0 * d["pp_tb"] - d["pp_vol"]
    for k in (1, 5, 15, 60):
        d[f"pp_ofi_{k}"] = (s.rolling(k, min_periods=1).sum()
                            / (d["pp_vol"].rolling(k, min_periods=1).sum() + 1e-12))
        n.append(f"pp_ofi_{k}")
    d["pp_moins_spot_ofi_5"] = d["pp_ofi_5"] - d["ofi_5"]
    part = np.log((d["pp_vol"] + 1e-9) / (d["bn_volume"] + 1e-9))
    d["pp_part_volume"] = part - part.rolling(1440, min_periods=360).mean()
    e = (d["pp_close"] / d["bn_close"] - 1.0) * 1e4
    d["pp_ecart_spot"] = e - e.rolling(60, min_periods=30).mean()
    e = (d["pp_close"] / d["close"] - 1.0) * 1e4
    d["pp_ecart_courtier"] = e - e.rolling(60, min_periods=30).mean()
    n += ["pp_moins_spot_ofi_5", "pp_part_volume", "pp_ecart_spot",
          "pp_ecart_courtier"]
    # --- PERPETUELS : prime sur l'indice ---
    p = d["prem"] * 1e4
    d["prem_bps"] = p
    d["prem_dev_60"] = p - p.rolling(60, min_periods=30).mean()
    d["prem_chg_5"] = p - p.shift(5)
    d["prem_rang"] = p.rolling(20_000, min_periods=2_000).rank(pct=True)
    n += ["prem_bps", "prem_dev_60", "prem_chg_5", "prem_rang"]
    # --- METRIQUES : positions ouvertes et ratios ---
    oi = np.log(d["sum_open_interest"].clip(lower=1e-9))
    r60 = np.log(d["close"]).diff(60)
    for k in (15, 60, 240):
        d[f"oi_chg_{k}"] = oi - oi.shift(k)
        n.append(f"oi_chg_{k}")
    # LE SENS DE L'OUVERTURE DE POSITIONS : des positions qui s'ouvrent
    # pendant que le prix monte sont des longs, pendant qu'il baisse des
    # shorts.
    d["oi_x_prix_60"] = d["oi_chg_60"] * np.sign(r60)
    d["taker_ls_5m"] = np.log(d["sum_taker_long_short_vol_ratio"].clip(lower=1e-6))
    ls = np.log(d["sum_toptrader_long_short_ratio"].clip(lower=1e-6))
    d["top_ls_chg_60"] = ls - ls.shift(60)
    g = np.log(d["count_long_short_ratio"].clip(lower=1e-6))
    d["foule_ls_chg_60"] = g - g.shift(60)
    n += ["oi_x_prix_60", "taker_ls_5m", "top_ls_chg_60", "foule_ls_chg_60"]
    # --- COINBASE : la demande americaine ---
    e = (d["cb_close"] / d["bn_close"] - 1.0) * 1e4
    d["prime_cb"] = e - e.rolling(1440, min_periods=360).mean()
    d["prime_cb_dev_60"] = e - e.rolling(60, min_periods=30).mean()
    d["prime_cb_chg_15"] = e - e.shift(15)
    part = np.log((d["cb_vol"] + 1e-9) / (d["bn_volume"] + 1e-9))
    d["cb_part_volume"] = part - part.rolling(1440, min_periods=360).mean()
    e = (d["cb_close"] / d["close"] - 1.0) * 1e4
    d["cb_ecart_courtier"] = e - e.rolling(60, min_periods=30).mean()
    n += ["prime_cb", "prime_cb_dev_60", "prime_cb_chg_15", "cb_part_volume",
          "cb_ecart_courtier"]
    return n


REFERENCES = ["ofi_5", "ofi_15", "ofi_60", "taker_ratio", "ecart_binance",
              "creux_rang", "flux_rang", "creux_x_flux", "returns",
              "close_ema_dev", "rend_jour", "vol_rel_creneau"]


def rendements(d: pd.DataFrame) -> dict:
    """Brut futur en bps : de l'ouverture de t+1 a celle de t+1+h."""
    o = d["open"].to_numpy(np.float64)
    out = {}
    for h in HORIZONS:
        r = np.full(len(o), np.nan)
        r[:-1 - h] = (o[1 + h:] / o[1:-h] - 1.0) * 1e4
        # RIEN NE DOIT LIRE LE TEST : un rendement qui finit au-dela de
        # FIN_MESURE est efface.
        r[FIN_MESURE - 1 - h:] = np.nan
        out[h] = r
    return out


def ic(x: np.ndarray, y: np.ndarray) -> tuple:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 500:
        return np.nan, np.nan, 0
    a = pd.Series(x[ok]).rank().to_numpy()
    b = pd.Series(y[ok]).rank().to_numpy()
    r = float(np.corrcoef(a, b)[0, 1])
    return r, r * np.sqrt(ok.sum()), int(ok.sum())


def deciles(x_app, x_mes, y_mes) -> float:
    """(decile haut - decile bas) / 2 sur la mesure, seuils pris a
    l'APPRENTISSAGE : le brut par trade d'une regle long/short aux extremes."""
    ok = np.isfinite(x_app)
    if ok.sum() < 1000:
        return np.nan
    bas, haut = np.quantile(x_app[ok], [0.1, 0.9])
    m = np.isfinite(x_mes) & np.isfinite(y_mes)
    h = y_mes[m & (x_mes >= haut)]
    b = y_mes[m & (x_mes <= bas)]
    if len(h) < 50 or len(b) < 50:
        return np.nan
    return float((h.mean() - b.mean()) / 2.0)


def mesure_colonnes(d, noms, R, titre):
    print(f"\n{titre}")
    print("  %-22s" % "colonne" + "".join(
        f"   h={h:<3d} IC app / IC mes (t)  " for h in (5, 15)) + "  extremes 15m")
    lignes = []
    for c in noms:
        x = d[c].to_numpy(np.float64)
        cel = []
        for h in HORIZONS:
            y = R[h]
            idx = np.arange(0, FIN_MESURE, h)
            ia = idx[idx < FIN_APPRENTISSAGE]
            im = idx[idx >= FIN_APPRENTISSAGE]
            r_a, t_a, _ = ic(x[ia], y[ia])
            r_m, t_m, n_m = ic(x[im], y[im])
            cel.append((h, r_a, t_a, r_m, t_m, n_m))
        idx15 = np.arange(0, FIN_MESURE, 15)
        ex = deciles(x[idx15[idx15 < FIN_APPRENTISSAGE]],
                     x[idx15[idx15 >= FIN_APPRENTISSAGE]],
                     R[15][idx15[idx15 >= FIN_APPRENTISSAGE]])
        lignes.append((c, cel, ex))
        txt = "  %-22s" % c
        for h, r_a, t_a, r_m, t_m, n_m in cel:
            if h in (5, 15):
                txt += f"   {r_a:+.4f} / {r_m:+.4f} ({t_m:+5.1f}) "
                txt += ("  " if np.sign(r_a) == np.sign(r_m) else "!=")
        txt += f"   {ex:+6.2f} bps"
        print(txt, flush=True)
    return lignes


def apport_modele(d, nouvelles, R):
    import lightgbm as lgb
    h = 15
    y = R[h]
    pas_app = 5
    ia = np.arange(0, FIN_APPRENTISSAGE - 1 - h, pas_app)
    im = np.arange(FIN_APPRENTISSAGE, FIN_MESURE - 1 - h, h)
    ya = y[ia]
    lo, hi = np.nanquantile(ya, [0.01, 0.99])
    ya = np.clip(ya, lo, hi)
    print(f"\n3. APPORT A UN MODELE — LightGBM, rendement brut a {h} min, "
          f"{len(ia):,} lignes d'apprentissage, {len(im):,} points de mesure disjoints")
    res = {}
    for nom, cols in (("les 68 colonnes du run", list(FEATURE_COLS)),
                      ("68 + nouvelles sources", list(FEATURE_COLS) + nouvelles),
                      ("nouvelles sources seules", nouvelles)):
        X = d[cols].to_numpy(np.float32)
        ok = np.isfinite(ya)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03,
                              num_leaves=31, min_child_samples=400,
                              subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=5.0,
                              n_jobs=4, verbose=-1)
        m.fit(X[ia][ok], ya[ok])
        p = m.predict(X[im])
        ym = y[im]
        r, t, n = ic(p, ym)
        ok_m = np.isfinite(ym)
        pm, yv = p[ok_m], ym[ok_m]
        txt = f"  {nom:26s} IC {r:+.4f} (t {t:+.1f})"
        for q in (0.10, 0.02):
            bas, haut = np.quantile(pm, [q, 1 - q])
            g = (yv[pm >= haut].mean() - yv[pm <= bas].mean()) / 2.0
            txt += f"   extremes {int(100*q)}% : {g:+.2f} bps/trade"
        print(txt, flush=True)
        res[nom] = (r, t)
        if nom == "68 + nouvelles sources":
            imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
            imp = imp / imp.sum()
            print("    part du gain des nouvelles colonnes : "
                  f"{100*imp[nouvelles].sum():.1f} %  |  les 8 premieres : "
                  + ", ".join(f"{k} {100*v:.1f}%" for k, v in
                              imp.sort_values(ascending=False).head(8).items()))
    return res


def main() -> int:
    d = charge()
    print(f"{len(d):,} minutes ; apprentissage [0 : {FIN_APPRENTISSAGE:,}) "
          f"{d['time'].iloc[0]} -> {d['time'].iloc[FIN_APPRENTISSAGE - 1]}, "
          f"mesure -> {d['time'].iloc[FIN_MESURE - 1]}. Le test n'est pas lu.")
    verifie_alignement(d.iloc[:FIN_MESURE])
    nouvelles = candidates(d)
    R = rendements(d)
    print(f"\ncout a battre : {COUT_BPS} bps par trade (validation du run)")
    print("IC = correlation de rang avec le rendement brut futur ; "
          "'!=' : le signe change entre les deux fenetres")
    mesure_colonnes(d, nouvelles, R, "1-2. LES NOUVELLES SOURCES")
    mesure_colonnes(d, REFERENCES, R, "   REFERENCES — colonnes deja dans le run")
    print("\n  IC a 1 et 60 minutes (mesure seulement), pour les nouvelles :")
    for c in nouvelles:
        x = d[c].to_numpy(np.float64)
        txt = f"    {c:22s}"
        for h in (1, 60):
            idx = np.arange(FIN_APPRENTISSAGE, FIN_MESURE, h)
            r, t, _ = ic(x[idx], R[h][idx])
            txt += f"   h={h:<2d} {r:+.4f} ({t:+5.1f})"
        print(txt)
    apport_modele(d, nouvelles, R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
