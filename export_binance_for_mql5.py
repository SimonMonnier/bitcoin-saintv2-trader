"""Exporte les features Binance vers un fichier binaire lisible par l'EA MQL5.

POURQUOI UN FICHIER, ET PAS UN APPEL RESEAU.
`WebRequest()` est totalement desactive dans le Strategy Tester MT5. L'EA ne
pourra JAMAIS interroger Binance pendant un backtest. La seule voie praticable
est de pre-exporter les features et de les lire depuis le disque -- et comme
l'EA doit se comporter identiquement en tester et en live, il lit ce fichier
dans les deux cas, quitte a le rafraichir regulierement en production.

FORMAT (little-endian, c'est celui de MQL5 sur x86)
    int64  magic      = 0x424E4331 ('BNC1')
    int64  t_debut     horodatage de la premiere minute, HEURE BROKER
    int32  n_minutes   nombre de lignes
    int32  n_colonnes  nombre de features par ligne
    puis n_minutes * n_colonnes float32, ligne par ligne

La grille est strictement minute, donc l'EA indexe en O(1) :
    ligne = (heure_bougie - t_debut) / 60
Les rares minutes absentes (bascules d'heure d'ete, une panne Binance de 9 h en
fevrier 2024) sont comblees par report de la derniere valeur connue : sans cela
l'indexation directe serait fausse. Le report reste borne par construction dans
build_binance_features.

    python export_binance_for_mql5.py
"""

import os
import struct
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saint_core import FEATURE_COLS_BINANCE, load_binance_cache

SYMBOL = "BTCUSD"
OUT_NAME = f"binance_{SYMBOL}.bin"
MAGIC = 0x424E4331


def dossier_commun_mt5():
    """<Terminal>\\Common\\Files — le seul visible depuis le Strategy Tester."""
    base = os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes",
                        "Terminal", "Common", "Files")
    return base if os.path.isdir(os.path.dirname(base)) else None


def main() -> int:
    df = load_binance_cache(SYMBOL)
    if df is None:
        print("Cache Binance absent. Lancer d'abord build_binance_features.py")
        return 1

    df = df[FEATURE_COLS_BINANCE].sort_index()
    print(f"Source : {len(df):,} minutes, {df.index[0]} -> {df.index[-1]}")

    # Grille continue : l'EA indexe par position, il ne peut pas chercher.
    grille = pd.date_range(df.index[0], df.index[-1], freq="1min")
    manquantes = len(grille) - len(df)
    plein = df.reindex(grille).ffill()
    if plein.isna().any().any():
        print("ERREUR : NaN residuels apres report — le cache est incomplet.")
        return 1
    print(f"Grille  : {len(plein):,} minutes "
          f"({manquantes} comblees par report, "
          f"{100*manquantes/len(plein):.4f} %)")

    t0 = int(plein.index[0].timestamp())
    donnees = plein.to_numpy(dtype="<f4")

    entete = struct.pack("<qqii", MAGIC, t0, len(plein), donnees.shape[1])
    cibles = [OUT_NAME]
    commun = dossier_commun_mt5()
    if commun:
        os.makedirs(commun, exist_ok=True)
        cibles.append(os.path.join(commun, OUT_NAME))

    for chemin in cibles:
        with open(chemin, "wb") as f:
            f.write(entete)
            f.write(donnees.tobytes())
        print(f"-> {chemin}  ({os.path.getsize(chemin)/1e6:.1f} Mo)")

    print()
    print(f"Colonnes, dans CET ordre (l'EA doit reprendre le meme) :")
    for i, c in enumerate(FEATURE_COLS_BINANCE):
        s = plein[c]
        print(f"   {i}  {c:<18} moy={s.mean():+.6f}  "
              f"min={s.min():+.6f}  max={s.max():+.6f}")
    print()
    print(f"t_debut = {t0} ({plein.index[0]}, heure broker)")
    if not commun:
        print("ATTENTION : dossier Common\\Files introuvable — copier "
              f"{OUT_NAME} a la main dans <Terminal>\\Common\\Files\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
