# -*- coding: utf-8 -*-
"""A QUEL NIVEAU ARMER LE BREAK-EVEN ?

CE QUE LE BREAK-EVEN FAIT, et ce qu'il coute. Des que le trade gagne `be` R,
le stop remonte au prix d'entree : la position ne peut plus rien perdre. Ca
parait gratuit, ca ne l'est pas. Un stop a l'entree est un stop TRES SERRE
compare au stop d'origine a 1 R, donc il se fait toucher par du bruit qui
n'aurait rien casse — et chaque trade ainsi coupe a zero est un trade qui ne
va plus chercher sa queue. Or c'est la queue qui paie : 14 % des trades
depassent +2 R et font 94 % des gains.

Le break-even echange donc de la variance contre de l'esperance, et la seule
question qui vaille est COMBIEN de chaque, a chaque niveau d'armement.

CE QUI EST MESURE. Le rendement moyen en R, le taux de reussite, et la part
des trades qui finissent exactement a zero — ceux que le break-even a coupes.
Armer trop tot se voit immediatement a cette derniere colonne.

POURQUOI CETTE MESURE EST FIABLE LA OU LES AUTRES NE LE SONT PAS. Elle ne
demande AUCUNE prediction : on rejoue la meme serie de prix sous des regles de
sortie differentes. Chaque barre est une observation, pas chaque trade. C'est
ce qui a permis de trancher la distance de trailing (-1.066 R sans, +1.791 R
a 4 R) la ou toutes les mesures predictives se noyaient dans le bruit.

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_break_even.py
"""
import os
import sys
import copy

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import cibles as C
import instruments as I
import training as T

PAS = 12
NIVEAUX = (None, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def main() -> int:
    cfg0 = T.PPOConfig()
    df = pd.read_pickle(I.INSTRUMENTS[cfg0.symbol]["cache"])
    n_fin = int(len(df) * 0.55) + int(len(df) * 0.15)
    idx = np.arange(cfg0.lookback + 1, n_fin - C.BORNE_DEFAUT - 2, PAS)

    r0 = _regle_courante(cfg0)
    print(f"{len(idx):,} occasions horaires sur les {n_fin:,} premieres barres")
    print(f"sortie en vigueur : stop {r0['sl']:g}xATR = 1 R | "
          f"trailing arme a {r0['ts']/r0['sl']:.1f} R, distance "
          f"{r0['td']/r0['sl']:.1f} R")
    print(f"La fenetre de test n'est pas lue.\n")

    # CE QU'ON ACHETE AVEC UN BREAK-EVEN, ce n'est pas du rendement moyen —
    # c'est de la PROTECTION. La table precedente ne mesurait que le premier.
    # On regarde donc aussi le cote perdant : la perte moyenne des trades
    # perdants, le dixieme percentile, et l'ecart-type. Si le break-even
    # reduit fortement ces trois-la, son cout en esperance peut se defendre.
    print(f"{'break-even':<12} {'R moyen':>9} {'vs sans':>9} {'a zero':>8} "
          f"{'perte moy':>10} {'p10':>8} {'ecart-type':>11} {'Sortino':>9}")
    print("-" * 82)
    ref = None
    for be in NIVEAUX:
        c = copy.copy(cfg0)
        c.atr_be_mult = 1e9 if be is None else be * cfg0.atr_sl_mult
        ra, _ = C.rendements(df, idx, c, indicateur=False)
        r = np.asarray(ra, float)
        r = r[np.isfinite(r)]
        if len(r) < 500:
            continue
        moy = float(r.mean())
        if ref is None:
            ref = moy
        nom = "aucun" if be is None else f"{be:g} R"
        # "a zero" : les trades que le break-even a coupes a l'entree. La
        # friction fait qu'ils finissent legerement negatifs, jamais pile 0.
        zero = float(np.mean(np.abs(r) < 0.12))
        perd = r[r < 0]
        neg = r.copy(); neg[neg > 0] = 0.0
        dd = float(np.sqrt(np.mean(neg ** 2)))
        print(f"{nom:<12} {moy:>+9.3f} {moy-ref:>+9.3f} {100*zero:>7.1f}% "
              f"{float(perd.mean()) if len(perd) else 0.0:>+10.3f} "
              f"{np.quantile(r, 0.10):>+8.2f} {float(r.std()):>11.3f} "
              f"{moy/max(dd, 1e-9):>+9.3f}")
    print("-" * 74)
    print("LECTURE. `Sortino` rapporte le rendement moyen au seul risque de")
    print("BAISSE : c'est exactement l'arbitrage qu'un break-even propose.")
    print("S'il ne monte a aucun niveau, la protection achetee coute plus")
    print("qu'elle ne rapporte, et le break-even ne se defend pas.")
    return 0


def _regle_courante(cfg):
    return C._regle(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
