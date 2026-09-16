"""Les memes bougies en direct que celles du jeu d'entrainement.

POURQUOI CE FICHIER EXISTE. Le live et l'entrainement construisaient leurs
colonnes par DEUX CHEMINS DIFFERENTS : `prepare_m5.construit` d'un cote,
`merge_m1_h1` de l'autre. Deux implementations de la meme chose, qui devaient
s'accorder par convention et que rien ne verifiait. Elles avaient d'ailleurs
cesse de s'accorder depuis longtemps — le live etait reste en M1, 30 colonnes,
stop a 2xATR, pendant que l'entrainement passait au M5, 260 colonnes et 8xATR.

Ici il n'y a plus qu'un chemin : on recupere les bougies brutes au MEME format
que `klines_5m_spot_BTCUSDT.pkl`, et on les passe a `prepare_m5.construit`. Les
colonnes du live sont alors identiques a celles de l'entrainement PAR
CONSTRUCTION, et `test_alignement.py` le verifie plutot que de le supposer.

POURQUOI BINANCE ET NON MT5. Quatre colonnes — part acheteuse agressive,
taille moyenne de trade, intensite — n'existent pas dans un flux CFD : MT5 ne
publie ni `taker_buy_base`, ni `quote_vol`, ni `nb_trades`. Le modele a appris
sur le spot Binance ; lui donner autre chose en production serait le faire lire
un marche qu'il n'a jamais vu. L'EXECUTION reste chez le courtier, et l'ecart
de prix entre les deux est exactement ce que la friction represente.

LA DERNIERE BOUGIE EST TOUJOURS JETEE. L'API rend la bougie EN FORMATION comme
derniere ligne. La garder injecterait jusqu'a cinq minutes de futur a chaque
cycle, sans qu'aucune erreur ne soit levee — la meme faute que le merge_asof
sans shift(1), payee une fois deja.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"
SYMBOLE = "BTCUSDT"
INTERVALLE = "5m"
LIMITE = 1000                  # maximum accepte par l'API

# Bougies d'echauffement a garder devant la premiere decision. Le bloc H4
# reclame FENETRE_EXT = 200 bougies H4, plus SENKOU_B = 52 et son decalage de
# 26 : ~280 bougies H4, soit 13 400 barres M5. On prend 16 000, comme
# test_causalite.py, pour que le live et le test parlent de la meme marge.
ECHAUFFEMENT = 16_000

COLONNES = ["time", "open", "high", "low", "close", "volume", "quote_vol",
            "nb_trades", "taker_buy_base"]


def _get(url: str, essais: int = 4):
    """GET JSON avec reprise : l'API rend 418/429 quand on insiste trop."""
    attente = 1.0
    for n in range(essais):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if n == essais - 1:
                raise
            time.sleep(attente)
            attente *= 2


def _en_trame(lignes) -> pd.DataFrame:
    """Du format brut de l'API aux neuf colonnes du jeu d'entrainement.

    L'ordre des champs est celui de la documentation : ouverture, OHLC,
    volume, fermeture, volume en quote, nombre de trades, achat agressif.
    """
    d = pd.DataFrame(lignes, columns=[
        "t_open", "open", "high", "low", "close", "volume", "t_close",
        "quote_vol", "nb_trades", "taker_buy_base", "taker_buy_quote", "_"])
    d["time"] = pd.to_datetime(d["t_open"].astype("int64"), unit="ms")
    for c in ("open", "high", "low", "close", "volume", "quote_vol",
              "taker_buy_base"):
        d[c] = d[c].astype(np.float64)
    d["nb_trades"] = d["nb_trades"].astype(np.int64)
    return d[COLONNES]


def bougies(depuis_ms: int | None = None, n: int = ECHAUFFEMENT) -> pd.DataFrame:
    """Les `n` dernieres bougies M5 CLOSES, format du jeu d'entrainement.

    `depuis_ms` permet de ne demander que la suite d'un historique deja en
    memoire — un seul appel par cycle de cinq minutes au lieu de seize.
    """
    morceaux = []
    if depuis_ms is not None:
        lignes = _get(f"{BASE}?symbol={SYMBOLE}&interval={INTERVALLE}"
                      f"&startTime={depuis_ms}&limit={LIMITE}")
        morceaux.append(_en_trame(lignes))
    else:
        # On remonte par paquets de 1 000 en partant de maintenant.
        fin = int(datetime.now(timezone.utc).timestamp() * 1000)
        restant = n
        while restant > 0:
            lignes = _get(f"{BASE}?symbol={SYMBOLE}&interval={INTERVALLE}"
                          f"&endTime={fin}&limit={min(LIMITE, restant)}")
            if not lignes:
                break
            t = _en_trame(lignes)
            morceaux.append(t)
            restant -= len(t)
            fin = int(t["time"].iloc[0].timestamp() * 1000) - 1
            time.sleep(0.12)      # on reste loin des quotas
    d = pd.concat(morceaux, ignore_index=True)
    d = d.drop_duplicates(subset="time").sort_values("time")
    return d.reset_index(drop=True)


def ferme(d: pd.DataFrame) -> pd.DataFrame:
    """Retire la bougie EN FORMATION, toujours rendue en derniere position.

    Une bougie de cinq minutes est close quand l'instant courant a depasse son
    ouverture de cinq minutes. Tester la duree plutot que faire confiance a la
    position protege aussi du cas ou l'API rend une bougie de retard.
    """
    if d.empty:
        return d
    maintenant = pd.Timestamp.utcnow().tz_localize(None)
    return d[d["time"] + pd.Timedelta(minutes=5) <= maintenant].reset_index(
        drop=True)


class Historique:
    """Garde les bougies entre deux cycles et ne demande que la suite.

    Un cycle de decision dure cinq minutes ; recharger seize mille bougies a
    chaque fois couterait seize appels pour une ligne nouvelle.
    """

    def __init__(self, n: int = ECHAUFFEMENT):
        self.n = n
        self.d = ferme(bougies(n=n))

    def actualise(self) -> pd.DataFrame:
        depuis = int(self.d["time"].iloc[-1].timestamp() * 1000) + 1
        suite = ferme(bougies(depuis_ms=depuis))
        if not suite.empty:
            self.d = pd.concat([self.d, suite], ignore_index=True)
            self.d = self.d.drop_duplicates(subset="time").tail(self.n)
            self.d = self.d.reset_index(drop=True)
        return self.d


def main() -> int:
    d = ferme(bougies(n=2000))
    print(f"{len(d):,} bougies M5 closes  "
          f"{d['time'].iloc[0]} -> {d['time'].iloc[-1]} (UTC)")
    print(d.tail(3).to_string(index=False))
    ecart = (pd.Timestamp.utcnow().tz_localize(None)
             - d["time"].iloc[-1]).total_seconds() / 60
    print(f"\nderniere bougie close il y a {ecart:.1f} min "
          f"(doit rester entre 5 et 10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
