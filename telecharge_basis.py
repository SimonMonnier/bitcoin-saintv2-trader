"""Basis perpetuel / spot a la MINUTE — telechargement et derivation.

POURQUOI CE CANDIDAT. Cinq essais d'ajout ont echoue cette nuit (Ichimoku,
H4, M5, M15, et le retrait du H1). Tous partageaient un defaut : ce sont des
AGREGATIONS de la meme serie de prix M1, deja presente. Elles n'ajoutent pas
d'information, seulement des colonnes dont il faut estimer les coefficients.

Les deux seules colonnes indispensables des trente — taker_ratio et
taker_1m_ma5 — viennent d'une SOURCE differente : le flux agressif de Binance.
Le basis est le dernier candidat de cette famille que je n'aie pas mesure :
l'ecart entre le prix du perpetuel et celui du spot mesure la demande de
levier, ce qu'aucune serie de prix ne contient.

Il varie a la MINUTE, contrairement au funding qui change toutes les 10 h et
qui vaut 0.0000 en apport mesure.

    python telecharge_basis.py
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
SORTIE = "binance_basis_BTCUSD.pkl"
I_OPEN_TIME, I_CLOSE = 0, 4


def _jour(marche: str, jour: dt.date):
    """marche : 'futures/um' (perpetuel) ou 'spot'."""
    url = (f"{VISION}/data/{marche}/daily/klines/{SYMBOL_BN}/1m/"
           f"{SYMBOL_BN}-1m-{jour:%Y-%m-%d}.zip")
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url, raw=True)))
    except Exception:
        return None
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    t = pd.to_numeric(df.iloc[:, I_OPEN_TIME], errors="coerce")
    df = df[t.notna()]
    if df.empty:
        return None
    t = pd.to_numeric(df.iloc[:, I_OPEN_TIME], errors="coerce")
    unite = "us" if t.iloc[0] > 1e14 else "ms"   # Binance a bascule ms -> us
    return pd.DataFrame({
        "time_utc": pd.to_datetime(t, unit=unite),
        "close": pd.to_numeric(df.iloc[:, I_CLOSE], errors="coerce").to_numpy(),
    })


def charge(marche: str, etiquette: str) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    f_cache = os.path.join(CACHE_DIR, f"close1m_{etiquette}.pkl")
    f_vides = os.path.join(CACHE_DIR, f"close1m_{etiquette}_absents.json")
    cache = pd.read_pickle(f_cache) if os.path.exists(f_cache) else pd.DataFrame()
    vides = set(json.load(open(f_vides))) if os.path.exists(f_vides) else set()
    deja = (set(pd.to_datetime(cache["time_utc"]).dt.strftime("%Y-%m-%d"))
            if not cache.empty else set())

    jours, j = [], DATE_DEBUT
    while j < DATE_FIN:
        cle = j.strftime("%Y-%m-%d")
        if cle not in deja and cle not in vides:
            jours.append(j)
        j += dt.timedelta(days=1)

    print(f"  {etiquette:<5} cache {len(deja)} jours, a telecharger {len(jours)}",
          flush=True)
    if jours:
        faits = [0]

        def trav(d):
            r = _jour(marche, d)
            faits[0] += 1
            if faits[0] % 200 == 0:
                print(f"    {etiquette} {faits[0]}/{len(jours)}", flush=True)
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
    return cache


def main() -> int:
    perp = charge("futures/um", "perp").set_index("time_utc")["close"]
    spot = charge("spot", "spot").set_index("time_utc")["close"]
    print(f"\nperp {len(perp):,} minutes  |  spot {len(spot):,} minutes")

    idx = perp.index.intersection(spot.index)
    p, s = perp.loc[idx].astype(float), spot.loc[idx].astype(float)
    print(f"intersection : {len(idx):,} minutes  "
          f"{idx.min()} -> {idx.max()}")

    out = pd.DataFrame(index=idx)
    # Prime du perpetuel sur le spot, en fraction. C'est la grandeur brute :
    # positive quand le levier long paie pour etre expose.
    base = (p - s) / s
    out["basis"] = base
    # Lissage causal sur 5 minutes — la fenetre qui a fonctionne pour le flux
    # taker. On teste les deux, le lissage pouvant etre ce qui fait la valeur.
    out["basis_ma5"] = base.rolling(5, min_periods=1).mean()
    # Variation : un basis qui S'ECARTE n'est pas un basis eleve.
    out["basis_chg"] = base.diff()
    # Prime rapportee a sa normale d'une journee : brute, elle encoderait le
    # REGIME de financement plutot que l'etat courant.
    m = base.rolling(1440, min_periods=60).mean()
    e = base.rolling(1440, min_periods=60).std()
    out["basis_z"] = (base - m) / (e + 1e-12)

    out = out.replace([np.inf, -np.inf], np.nan)
    out.to_pickle(SORTIE)
    print(f"\n{SORTIE} ecrit : {out.shape}")
    print(f"{'colonne':<12} {'non-NaN':>9} {'autocorr 1min':>14} "
          f"{'mediane':>12}")
    print("-" * 52)
    for c in out.columns:
        v = out[c].dropna().to_numpy()
        print(f"{c:<12} {100*out[c].notna().mean():8.2f}% "
              f"{np.corrcoef(v[:-1], v[1:])[0,1]:14.4f} {np.median(v):12.6f}")
    print("\nrepere : taker_1m_ma5 0.86 (apport +0.0087) | "
          "funding 0.9996 (apport 0.0000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
