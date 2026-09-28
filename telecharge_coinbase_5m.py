# -*- coding: utf-8 -*-
"""Bougies 5 minutes BTC-USD de Coinbase, d'aout 2017 a septembre 2026.

2026-09-28, demande du proprietaire : passer le jeu en M5, parties d'une
journee, avec les donnees 5 minutes de Binance et de Coinbase que MT5 n'a pas
sur une telle profondeur. Le cache Coinbase existant (1 minute) ne commence
qu'en octobre 2024 ; celui-ci couvre la meme periode que Binance, pour que
l'ecart de prix Coinbase / Binance soit une colonne sur neuf ans.

API publique, 300 bougies par requete (25 heures en 5 minutes), une dizaine
de requetes par seconde toleree : on reste sous trois. Reprise possible :
les lots deja telecharges sont gardes dans un fichier partiel.

    python telecharge_coinbase_5m.py
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.request

import pandas as pd

CACHE = "cache_coinbase_5m_BTCUSD.pkl"
PARTIEL = "cache_coinbase_5m_BTCUSD.partiel.pkl"
DEBUT = dt.datetime(2017, 8, 1)
FIN = dt.datetime(2026, 9, 15)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    lignes = []
    t = DEBUT
    if os.path.exists(PARTIEL):
        d0 = pd.read_pickle(PARTIEL)
        lignes = d0.values.tolist()
        t = pd.Timestamp(d0["t"].max(), unit="s").to_pydatetime() + dt.timedelta(minutes=5)
        print(f"reprise a {t} ({len(lignes):,} bougies deja la)", flush=True)
    pas = dt.timedelta(minutes=5 * 300)
    n = 0
    while t < FIN:
        b = min(t + pas, FIN)
        u = ("https://api.exchange.coinbase.com/products/BTC-USD/candles?"
             f"granularity=300&start={t:%Y-%m-%dT%H:%M:%SZ}&end={b:%Y-%m-%dT%H:%M:%SZ}")
        lot = []
        for essai in range(6):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "kairos-mesure"})
                lot = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception:
                time.sleep(1 + 2 * essai)
        lignes += lot
        n += 1
        if n % 200 == 0:
            print(f"  {t:%Y-%m-%d}  {len(lignes):,} bougies", flush=True)
            pd.to_pickle(pd.DataFrame(lignes, columns=["t", "low", "high", "open", "close", "volume"]),
                         PARTIEL)
        t = b
        time.sleep(0.35)
    d = pd.DataFrame(lignes, columns=["t", "low", "high", "open", "close", "volume"])
    d["time"] = pd.to_datetime(d["t"], unit="s")
    d = d.drop(columns=["t"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    d.to_pickle(CACHE)
    if os.path.exists(PARTIEL):
        os.remove(PARTIEL)
    print(f"ecrit {CACHE} : {len(d):,} bougies  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
