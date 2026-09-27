# -*- coding: utf-8 -*-
"""Aucune colonne du BTC M15 ne doit changer quand on coupe le futur.

La meme methode que `test_causalite_btc_m1.py`, sur les bougies de 15
minutes (2026-09-27) : chaque feature est calculee une fois sur la serie
entiere, une fois sur la serie TRONQUEE a la bougie t. Si la valeur en t
differe, le calcul a lu une bougie posterieure a t.

LES ENDROITS LES PLUS EXPOSES : les blocs H1 et H4, joints a la bougie en
formation sans le shift(1) ; les fenetres converties de minutes en bougies ;
l'horloge et les journees, lues a la DERNIERE minute de la bougie.

    python test_causalite_btc_m15.py
"""
import sys

import numpy as np
import pandas as pd

import prepare_btc_m1 as P1
import prepare_btc_m15 as P15

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ~104 jours de minutes : assez pour les 30 jours de `tend_mom_mois`.
DEBUT, LONGUEUR = 100_000, 150_000
brut = pd.read_pickle(P1.BRUT).iloc[DEBUT:DEBUT + LONGUEUR].reset_index(drop=True)
m15 = P15.agrege_m15(P1.joint_sources(brut))
n = len(m15)
POINTS = tuple(int(n * f) for f in (0.62, 0.70, 0.78, 0.86, 0.93)) + (n - 1,)

entier = P15.features_m15(m15)
cols = list(P15.FEATURE_COLS_M15)
absentes = [c for c in cols if c not in entier.columns]
print(f"{n} bougies M15 ; {len(cols) - len(absentes)} colonnes produites"
      + (f", ABSENTES : {absentes}" if absentes else ""))
if absentes:
    raise SystemExit(1)

ecarts, finies = [], 0
for t in POINTS:
    tronque = P15.features_m15(m15.iloc[:t + 1])
    a = entier.loc[t, cols].to_numpy(np.float64)
    b = tronque.loc[t, cols].to_numpy(np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    finies += int(fin.sum())
    diff = np.zeros(len(cols), bool)
    diff[fin] = np.abs(a[fin] - b[fin]) > 1e-9 * (1.0 + np.abs(a[fin]))
    diff |= np.isfinite(a) != np.isfinite(b)
    fautives = [cols[i] for i in np.flatnonzero(diff)]
    print(f"  t = {t:>6}  {len(fautives)} colonne(s) qui lisent le futur"
          + (f" : {fautives[:6]}" if fautives else ""))
    ecarts += fautives

print(f"\n{finies} valeurs comparees sur {len(POINTS) * len(cols)}")
if finies < 0.95 * len(POINTS) * len(cols):
    print("ECHEC : trop de valeurs non definies, l'amorce est trop courte")
    raise SystemExit(1)
if ecarts:
    print(f"ECHEC : {len(set(ecarts))} colonne(s) lisent le futur")
    raise SystemExit(1)
print("OK : aucune colonne ne change quand on coupe le futur")
