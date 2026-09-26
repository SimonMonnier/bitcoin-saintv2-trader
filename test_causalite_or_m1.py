# -*- coding: utf-8 -*-
"""Aucune colonne de l'or M1 ne doit changer quand on coupe le futur.

La meme methode que `test_causalite.py`, sur la chaine de l'or M1 et ses
contextes M5 et M15 (2026-09-26) : chaque feature est calculee une fois sur
la serie entiere, une fois sur la serie TRONQUEE a l'instant t. Si la valeur
en t differe, le calcul a lu une barre posterieure a t.

LES BLOCS M5 ET M15 SONT L'ENDROIT LE PLUS EXPOSE. `merge_asof` choisit la
bougie superieure qui CONTIENT la minute courante, donc une bougie encore en
formation : sans le shift(1), chaque minute recevrait jusqu'a quatorze minutes
de futur en M15, et aucune erreur ne serait levee.

    python test_causalite_or_m1.py
"""
import sys

import numpy as np
import pandas as pd

import prepare_m5
import prepare_or_m1 as P
from saint_core import FEATURE_COLS, SYMBOLE

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# L'OR N'EST PLUS ENTRAINE depuis le 2026-09-26 au soir : son jeu de
# colonnes n'est plus celui de `FEATURE_COLS`. Voir test_causalite_btc_m1.
if SYMBOLE != "XAUUSD":
    print(f"SAUTE : l'instrument entraine est {SYMBOLE}")
    raise SystemExit(0)

# Assez d'amorce pour `tend_mom_mois` (8 640 barres) et l'Ichimoku M15.
DEBUT, LONGUEUR = 200_000, 14_000
POINTS = (11_000, 11_517, 12_003, 12_871, 13_402, 13_999)

brut = pd.read_pickle(P.BRUT).iloc[DEBUT:DEBUT + LONGUEUR].reset_index(drop=True)


def features(df):
    d, _, _ = prepare_m5.construit(df.copy(), avec_flux=False, jour=P.JOUR_M1,
                                   echelles=P.ECHELLES_OR_M1)
    return d


entier = features(brut)
cols = [c for c in FEATURE_COLS if c in entier.columns]
print(f"{len(cols)} colonnes de FEATURE_COLS produites par la chaine")

ecarts = []
for t in POINTS:
    tronque = features(brut.iloc[:t + 1])
    a = entier.loc[t, cols].to_numpy(np.float64)
    b = tronque.loc[t, cols].to_numpy(np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    diff = np.zeros(len(cols), bool)
    diff[fin] = np.abs(a[fin] - b[fin]) > 1e-9 * (1.0 + np.abs(a[fin]))
    diff |= np.isfinite(a) != np.isfinite(b)
    fautives = [cols[i] for i in np.flatnonzero(diff)]
    print(f"  t = {t:>6}  {len(fautives)} colonne(s) qui lisent le futur"
          + (f" : {fautives[:6]}" if fautives else ""))
    ecarts += fautives

bloc = [c for c in cols if c.endswith(("_m5", "_m15")) or "_m5_" in c
        or "_m15_" in c]
print(f"\ndont {len(bloc)} colonnes des contextes M5 et M15")
if ecarts:
    print(f"ECHEC : {len(set(ecarts))} colonne(s) lisent le futur")
    raise SystemExit(1)
print("OK : aucune colonne ne change quand on coupe le futur")
