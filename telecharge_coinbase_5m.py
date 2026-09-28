# -*- coding: utf-8 -*-
"""Bougies 5 minutes BTC-USD de Coinbase, d'aout 2017 a septembre 2026.

2026-09-28, demande du proprietaire : passer le jeu en M5, parties d'une
journee, avec les donnees 5 minutes de Binance et de Coinbase que MT5 n'a pas
sur une telle profondeur. Le cache Coinbase existant (1 minute) ne commence
qu'en octobre 2024 ; celui-ci couvre la meme periode que Binance, pour que
l'ecart de prix Coinbase / Binance soit une colonne sur neuf ans.

API publique, 300 bougies par requete (25 heures en 5 minutes), une dizaine
de requetes par seconde toleree. Reprise possible : les lots deja
telecharges sont gardes dans un fichier partiel, et l'instant atteint dans
un fichier `.ou` (une crypto listee apres 2017 n'a pas de bougie a
enregistrer avant sa cotation).

EN PARALLELE depuis le 2026-09-28 : une requete a la fois laissait ~1 lot
par seconde (neuf ans d'ETH : plus d'une heure). `FILS` requetes en vol,
les lots toujours enregistres dans l'ordre.

    python telecharge_coinbase_5m.py [ETH-USD]

AUTRES CRYPTOS — 2026-09-28 : le produit Coinbase en argument (BTC-USD par
defaut), le cache s'appelle alors cache_coinbase_5m_<ETHUSD>.pkl.
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

PRODUIT = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
CACHE = f"cache_coinbase_5m_{PRODUIT.replace('-', '')}.pkl"
PARTIEL = f"cache_coinbase_5m_{PRODUIT.replace('-', '')}.partiel.pkl"
OU = PARTIEL.replace(".pkl", ".ou")
FILS = 3
DEBUT = dt.datetime(2017, 8, 1)
FIN = dt.datetime(2026, 9, 15)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _lot(fenetre):
    t, b = fenetre
    u = (f"https://api.exchange.coinbase.com/products/{PRODUIT}/candles?"
         f"granularity=300&start={t:%Y-%m-%dT%H:%M:%SZ}&end={b:%Y-%m-%dT%H:%M:%SZ}")
    for essai in range(8):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "kairos-mesure"})
            lot = json.loads(urllib.request.urlopen(req, timeout=60).read())
            time.sleep(0.35)
            return lot, True
        except Exception:
            time.sleep(1 + 2 * essai)
    return [], False


def main() -> int:
    cols = ["t", "low", "high", "open", "close", "volume"]
    lignes = []
    t = DEBUT
    if os.path.exists(PARTIEL):
        d0 = pd.read_pickle(PARTIEL)
        lignes = d0.values.tolist()
        if os.path.exists(OU):
            t = dt.datetime.fromisoformat(open(OU).read().strip())
        elif len(d0):
            t = pd.Timestamp(d0["t"].max(), unit="s").to_pydatetime() + dt.timedelta(minutes=5)
        print(f"reprise a {t} ({len(lignes):,} bougies deja la)", flush=True)
    pas = dt.timedelta(minutes=5 * 300)
    fen = []
    while t < FIN:
        fen.append((t, min(t + pas, FIN)))
        t = fen[-1][1]
    rates = 0
    with ThreadPoolExecutor(FILS) as ex:
        for i0 in range(0, len(fen), 200):
            morceau = fen[i0:i0 + 200]
            for lot, ok in ex.map(_lot, morceau):
                lignes += lot
                rates += not ok
            pd.to_pickle(pd.DataFrame(lignes, columns=cols), PARTIEL)
            with open(OU, "w") as fh:
                fh.write(morceau[-1][1].isoformat())
            print(f"  {PRODUIT} {morceau[-1][1]:%Y-%m-%d}  {len(lignes):,} bougies"
                  + (f"  ({rates} lots en echec)" if rates else ""), flush=True)
    d = pd.DataFrame(lignes, columns=cols)
    d["time"] = pd.to_datetime(d["t"], unit="s")
    d = d.drop(columns=["t"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    d.to_pickle(CACHE)
    for f in (PARTIEL, OU):
        if os.path.exists(f):
            os.remove(f)
    print(f"ecrit {CACHE} : {len(d):,} bougies  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}"
          + (f"  ATTENTION {rates} lots de 25 h perdus apres 8 essais" if rates else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
