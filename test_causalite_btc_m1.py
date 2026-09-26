# -*- coding: utf-8 -*-
"""Aucune colonne du scalping BTC M1 ne doit changer quand on coupe le futur.

La meme methode que `test_causalite_or_m1.py`, sur la chaine du BTC M1
(2026-09-26) : contextes M5 et M15, et les familles de `features_scalping`
— horloge, regime rapporte a son creneau, flux Binance, ancres de prix,
rangs glissants. Chaque feature est calculee une fois sur la serie entiere,
une fois sur la serie TRONQUEE a l'instant t. Si la valeur en t differe, le
calcul a lu une barre posterieure a t.

LES ENDROITS LES PLUS EXPOSES :
  - les blocs M5 et M15, que `merge_asof` joint a la bougie en formation
    sans le shift(1) ;
  - les normes de creneau, qui incluraient la minute courante sans leur
    decalage dans le groupe ;
  - les extremes du jour et de la veille, qui liraient la journee entiere
    si on les calculait par `transform("max")`.

    python test_causalite_btc_m1.py
"""
import sys

import numpy as np
import pandas as pd

import features_scalping as FS
import prepare_btc_m1 as P
import prepare_m5
from saint_core import FEATURE_COLS, SYMBOLE

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if SYMBOLE != "BTCUSD":
    print(f"SAUTE : l'instrument entraine est {SYMBOLE}")
    raise SystemExit(0)

# Assez d'amorce pour `tend_mom_mois` (30 jours = 43 200 barres M1).
DEBUT, LONGUEUR = 100_000, 62_000
POINTS = (50_000, 52_517, 55_003, 57_871, 60_402, 61_999)

brut = pd.read_pickle(P.BRUT).iloc[DEBUT:DEBUT + LONGUEUR].reset_index(drop=True)
bn = FS.aligne_binance(pd.read_pickle(P.CACHE_BINANCE))
brut = brut.merge(bn, on="time", how="left")


def features(df):
    d, _, _ = prepare_m5.construit(df.copy(), avec_flux=False, jour=P.JOUR_M1,
                                   echelles=P.ECHELLES_BTC_M1,
                                   avec_structures=False)
    return FS.ajoute(d)


entier = features(brut)
cols = [c for c in FEATURE_COLS if c in entier.columns]
absentes = [c for c in FEATURE_COLS if c not in entier.columns]
print(f"{len(cols)} colonnes de FEATURE_COLS produites par la chaine"
      + (f", ABSENTES : {absentes}" if absentes else ""))
if absentes:
    raise SystemExit(1)

ecarts, finies = [], 0
for t in POINTS:
    tronque = features(brut.iloc[:t + 1])
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

# UN TEST QUI NE COMPARE QUE DES NaN NE PROUVE RIEN : il faut que les
# points tombent apres l'amorce.
print(f"\n{finies} valeurs comparees sur {len(POINTS) * len(cols)}")
if finies < 0.95 * len(POINTS) * len(cols):
    print("ECHEC : trop de valeurs non definies, l'amorce est trop courte")
    raise SystemExit(1)
if ecarts:
    print(f"ECHEC : {len(set(ecarts))} colonne(s) lisent le futur")
    raise SystemExit(1)
print("OK : aucune colonne ne change quand on coupe le futur")
