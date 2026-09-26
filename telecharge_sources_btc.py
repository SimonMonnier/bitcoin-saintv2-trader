# -*- coding: utf-8 -*-
"""Les sources BTC gratuites que le projet n'avait pas encore — telechargement.

2026-09-26, demande du proprietaire : mesurer si d'autres sources gratuites
que le spot Binance et MT5 portent de la direction. Trois familles, toutes
avec deux ans d'historique a la minute :

  PERPETUELS BINANCE (USDT-M), sur data.binance.vision
    klines 1 min          le flux agressif sur le marche ou se fait la
                          majorite du volume
    premiumIndexKlines    l'ecart perpetuel / indice spot, a la minute
    metrics (5 min)       positions ouvertes, ratios longs/shorts, ratio
                          acheteurs/vendeurs agressifs
  COINBASE BTC-USD        bougies 1 min par l'API publique : la demande
                          americaine, en dollars et pas en USDT

Tout est en UTC. L'alignement sur l'heure du serveur se fait a la mesure,
par `features_scalping.serveur_depuis_utc`.

    python telecharge_sources_btc.py
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEBUT = dt.date(2024, 10, 1)
FIN = dt.date(2026, 9, 21)
BASE = "https://data.binance.vision/data/futures/um"
CACHES = {
    "perp": "cache_perp_1m_BTCUSDT.pkl",
    "premium": "cache_premium_1m_BTCUSDT.pkl",
    "metrics": "cache_metrics_5m_BTCUSDT.pkl",
    "coinbase": "cache_coinbase_1m_BTCUSD.pkl",
}
NOMS_KLINES = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "count", "taker_buy_volume",
               "taker_buy_quote_volume", "ignore"]


def _zip_csv(url: str, noms=None) -> pd.DataFrame | None:
    for essai in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=90).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            d = pd.read_csv(z.open(z.namelist()[0]), header=None, names=noms,
                            low_memory=False)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * essai)
        except Exception:
            time.sleep(2 + 3 * essai)
    return None


def _horodatage(t: pd.Series) -> pd.Series:
    """Millisecondes ou microsecondes, tranche LIGNE PAR LIGNE. Voir
    `prepare_btc_m1.charge_binance_1m`."""
    t = t.astype("int64")
    micro = t > int(1e14)
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    out[micro] = pd.to_datetime(t[micro], unit="us")
    out[~micro] = pd.to_datetime(t[~micro], unit="ms")
    return out


def klines(jeu: str) -> pd.DataFrame:
    """Klines 1 min des perpetuels : archives mensuelles, puis journalieres
    pour le mois pas encore publie."""
    urls = []
    m = dt.date(DEBUT.year, DEBUT.month, 1)
    while m <= FIN:
        urls.append(f"{BASE}/monthly/{jeu}/BTCUSDT/1m/BTCUSDT-1m-{m:%Y-%m}.zip")
        m = (m + dt.timedelta(days=32)).replace(day=1)
    with ThreadPoolExecutor(6) as ex:
        lots = list(ex.map(lambda u: (u, _zip_csv(u, NOMS_KLINES)), urls))
    morceaux = []
    for u, d in lots:
        if d is not None:
            morceaux.append(d)
            continue
        # LE MOIS COURANT n'a pas encore d'archive mensuelle.
        mois = u[-11:-4]
        j = dt.date(int(mois[:4]), int(mois[5:]), 1)
        jours = []
        while j <= FIN and f"{j:%Y-%m}" == mois:
            jours.append(f"{BASE}/daily/{jeu}/BTCUSDT/1m/BTCUSDT-1m-{j:%Y-%m-%d}.zip")
            j += dt.timedelta(days=1)
        with ThreadPoolExecutor(6) as ex:
            morceaux += [x for x in ex.map(lambda v: _zip_csv(v, NOMS_KLINES), jours)
                         if x is not None]
        print(f"    {jeu} {mois} : {len(jours)} archives journalieres", flush=True)
    d = pd.concat(morceaux, ignore_index=True)
    d = d[pd.to_numeric(d["open_time"], errors="coerce").notna()].copy()
    for c in NOMS_KLINES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = _horodatage(d["open_time"])
    d = d.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    print(f"  {jeu} : {len(d):,} minutes  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}",
          flush=True)
    return d


def metrics() -> pd.DataFrame:
    jours, j = [], DEBUT
    while j <= FIN:
        jours.append(f"{BASE}/daily/metrics/BTCUSDT/BTCUSDT-metrics-{j:%Y-%m-%d}.zip")
        j += dt.timedelta(days=1)
    with ThreadPoolExecutor(8) as ex:
        lots = [x for x in ex.map(_zip_csv, jours) if x is not None]
    d = pd.concat(lots, ignore_index=True)
    # Chaque fichier journalier porte sa ligne d'en-tete.
    d = d[d.iloc[:, 0].astype(str) != "create_time"].copy()
    d.columns = ["create_time", "symbol", "sum_open_interest",
                 "sum_open_interest_value", "count_toptrader_long_short_ratio",
                 "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                 "sum_taker_long_short_vol_ratio"][:d.shape[1]]
    d["time"] = pd.to_datetime(d["create_time"])
    for c in d.columns[2:]:
        if c != "time":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop(columns=["create_time", "symbol"]).drop_duplicates("time")
    d = d.sort_values("time").reset_index(drop=True)
    print(f"  metrics : {len(d):,} lignes de 5 min  {d['time'].iloc[0]} -> "
          f"{d['time'].iloc[-1]}  ({len(lots)} jours sur {len(jours)})", flush=True)
    return d


def coinbase() -> pd.DataFrame:
    """300 bougies par requete, l'API publique en tolere une dizaine par
    seconde : on reste sous trois."""
    t0 = dt.datetime(DEBUT.year, DEBUT.month, DEBUT.day)
    t_fin = dt.datetime(FIN.year, FIN.month, FIN.day) + dt.timedelta(days=1)
    pas = dt.timedelta(minutes=300)
    lignes, n, t = [], 0, t0
    while t < t_fin:
        b = min(t + pas, t_fin)
        u = ("https://api.exchange.coinbase.com/products/BTC-USD/candles?"
             f"granularity=60&start={t:%Y-%m-%dT%H:%M:%SZ}&end={b:%Y-%m-%dT%H:%M:%SZ}")
        for essai in range(6):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "kairos-mesure"})
                lot = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception:
                time.sleep(1 + 2 * essai)
                lot = []
        lignes += lot
        n += 1
        if n % 300 == 0:
            print(f"    coinbase {t:%Y-%m-%d}  {len(lignes):,} minutes", flush=True)
        t = b
        time.sleep(0.35)
    d = pd.DataFrame(lignes, columns=["t", "low", "high", "open", "close", "volume"])
    d["time"] = pd.to_datetime(d["t"], unit="s")
    d = d.drop(columns=["t"]).drop_duplicates("time").sort_values("time")
    d = d.reset_index(drop=True)
    print(f"  coinbase : {len(d):,} minutes  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}",
          flush=True)
    return d


def main() -> int:
    for nom, f in (("perp", lambda: klines("klines")),
                   ("premium", lambda: klines("premiumIndexKlines")),
                   ("metrics", metrics),
                   ("coinbase", coinbase)):
        if os.path.exists(CACHES[nom]):
            print(f"  {nom} : deja en cache ({CACHES[nom]})", flush=True)
            continue
        t = time.time()
        d = f()
        d.to_pickle(CACHES[nom])
        print(f"  -> {CACHES[nom]} ({time.time() - t:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
