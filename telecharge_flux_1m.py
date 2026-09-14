"""Telecharge les klines Binance 1 MINUTE et en derive des features de FLUX.

POURQUOI. Mesure du 2026-09-14 : sur les six colonnes de
binance_features_BTCUSD.pkl, `taker_ratio` vaut a elle seule +0.0498 d'AUC
(0.6178 -> 0.5680 si on la retire) et `ls_ratio_top` vaut +0.0000. Les quatre
inutilisees valent 0.4847 en AUC a elles seules, soit moins que le hasard.

Ce qui separe les deux groupes est leur VITESSE. Autocorrelation a 1 minute :
taker_ratio 0.82, tout le reste 0.99 a 0.9999. Le funding change toutes les
10 heures ; sur un trade de 7 minutes c'est une constante, donc elle ne peut
pas departager deux entrees espacees de dix minutes.

Or le pipeline n'a jamais consomme les archives `klines`, qui portent le flux
a la MINUTE : volume reel, nombre de trades, volume pris a l'achat au marche.
MetaTrader n'en fournit aucun — son `tick_volume` compte les changements de
prix, pas les montants echanges.

RESERVE. Une autocorrelation basse n'est pas en soi une qualite : du bruit pur
en a une nulle. La variabilite est une condition NECESSAIRE, pas suffisante.
Ce script ne fait que produire les colonnes ; c'est mesure_flux_1m.py qui
tranche, par apport marginal d'AUC.

    python telecharge_flux_1m.py
"""

import io
import os
import json
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from build_binance_features import _get, VISION, SYMBOL_BN, N_THREADS, CACHE_DIR

DATE_DEBUT = dt.date(2022, 12, 14)
DATE_FIN = dt.date(2026, 9, 14)
SORTIE = "binance_flux_1m_BTCUSD.pkl"

# Colonnes du kline futures UM, par position (le CSV n'a pas d'entete).
I_OPEN_TIME, I_VOLUME, I_QUOTE_VOL, I_NB_TRADES, I_TAKER_BUY_BASE = 0, 5, 7, 8, 9


def _jour(jour: dt.date):
    url = (f"{VISION}/data/futures/um/daily/klines/{SYMBOL_BN}/1m/"
           f"{SYMBOL_BN}-1m-{jour:%Y-%m-%d}.zip")
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url, raw=True)))
    except Exception:
        return None
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    # Certaines archives recentes portent une ligne d'entete ; on la jette.
    t = pd.to_numeric(df.iloc[:, I_OPEN_TIME], errors="coerce")
    df = df[t.notna()]
    if df.empty:
        return None
    t = pd.to_numeric(df.iloc[:, I_OPEN_TIME], errors="coerce")
    # Binance a bascule de ms a us dans les archives recentes.
    unite = "us" if t.iloc[0] > 1e14 else "ms"
    v   = pd.to_numeric(df.iloc[:, I_VOLUME], errors="coerce")
    qv  = pd.to_numeric(df.iloc[:, I_QUOTE_VOL], errors="coerce")
    nt  = pd.to_numeric(df.iloc[:, I_NB_TRADES], errors="coerce")
    tbb = pd.to_numeric(df.iloc[:, I_TAKER_BUY_BASE], errors="coerce")
    return pd.DataFrame({
        "time_utc": pd.to_datetime(t, unit=unite),
        "volume": v.to_numpy(), "quote_vol": qv.to_numpy(),
        "nb_trades": nt.to_numpy(), "taker_buy_base": tbb.to_numpy(),
    })


def main() -> int:
    os.makedirs(CACHE_DIR, exist_ok=True)
    f_cache = os.path.join(CACHE_DIR, "klines1m_utc.pkl")
    f_vides = os.path.join(CACHE_DIR, "klines1m_absents.json")

    cache = pd.read_pickle(f_cache) if os.path.exists(f_cache) else pd.DataFrame()
    vides = set(json.load(open(f_vides))) if os.path.exists(f_vides) else set()
    deja = set()
    if not cache.empty:
        deja = set(pd.to_datetime(cache["time_utc"]).dt.strftime("%Y-%m-%d"))

    jours, j = [], DATE_DEBUT
    while j < DATE_FIN:
        cle = j.strftime("%Y-%m-%d")
        if cle not in deja and cle not in vides:
            jours.append(j)
        j += dt.timedelta(days=1)

    print(f"cache : {len(deja)} jours presents, {len(vides)} absents connus")
    if jours:
        print(f"telechargement de {len(jours)} jours sur {N_THREADS} fils...")
        faits = [0]

        def trav(d):
            r = _jour(d)
            faits[0] += 1
            if faits[0] % 100 == 0:
                print(f"  {faits[0]}/{len(jours)} ({100*faits[0]/len(jours):.0f} %)",
                      flush=True)
            return d, r

        with ThreadPoolExecutor(N_THREADS) as ex:
            res = list(ex.map(trav, jours))
        neufs = [r for _, r in res if r is not None and len(r)]
        for d, r in res:
            if r is None or not len(r):
                vides.add(d.strftime("%Y-%m-%d"))
        if neufs:
            cache = pd.concat([cache] + neufs, ignore_index=True)
            cache = cache.drop_duplicates("time_utc").sort_values("time_utc")
            cache.to_pickle(f_cache)
        json.dump(sorted(vides), open(f_vides, "w"))

    if cache.empty:
        print("aucune donnee")
        return 1

    d = cache.set_index("time_utc").sort_index()
    print(f"\n{len(d):,} minutes  {d.index[0]} -> {d.index[-1]}")

    v  = d["volume"].clip(lower=1e-8)
    qv = d["quote_vol"].clip(lower=1e-8)
    nt = d["nb_trades"].clip(lower=1)
    part = (d["taker_buy_base"] / v).clip(1e-4, 1 - 1e-4)

    out = pd.DataFrame(index=d.index)
    # Meme forme que le taker_ratio 5 min en service : un log-odds, symetrique
    # autour de zero et non borne, plutot qu'une proportion dans [0,1].
    out["taker_1m"] = np.log(part / (1 - part))
    # Lissage causal sur 5 minutes : la version 5 min qui marche deja est un
    # agregat. On teste les deux, l'agregation pouvant etre ce qui fait sa
    # valeur plutot qu'un defaut.
    out["taker_1m_ma5"] = out["taker_1m"].rolling(5, min_periods=1).mean()
    # Grandeurs de niveau ramenees a leur regime recent : un volume brut est
    # non stationnaire sur quatre ans (le prix a quadruple).
    for nom, s in (("volume", v), ("nb_trades", nt)):
        lg = np.log(s)
        out[f"{nom}_z"] = ((lg - lg.rolling(1440, min_periods=60).mean())
                           / (lg.rolling(1440, min_periods=60).std() + 1e-8))
    lg_taille = np.log(qv / nt)
    out["taille_trade_z"] = ((lg_taille - lg_taille.rolling(1440, min_periods=60).mean())
                             / (lg_taille.rolling(1440, min_periods=60).std() + 1e-8))

    out = out.replace([np.inf, -np.inf], np.nan)
    out.to_pickle(SORTIE)
    print(f"\n{SORTIE} ecrit : {out.shape}")
    print(f"{'colonne':<18} {'non-NaN':>9} {'autocorr 1min':>14}")
    print("-" * 44)
    for c in out.columns:
        s = out[c].dropna().to_numpy()
        print(f"{c:<18} {100*out[c].notna().mean():8.2f}% "
              f"{np.corrcoef(s[:-1], s[1:])[0,1]:14.4f}")
    print("\nIndex en UTC — l'alignement sur l'heure du courtier est fait par "
          "mesure_flux_1m.py, par recalage sur le taker_ratio deja en service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
