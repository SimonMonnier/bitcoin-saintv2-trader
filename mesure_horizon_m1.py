# -*- coding: utf-8 -*-
"""Sur quel horizon un trade M1 peut-il payer son passage, et combien ?

LE PEAGE EST FIXE, LE MOUVEMENT GRANDIT AVEC LE TEMPS. Un aller-retour coute
1.36 point de base sur l'or — deux fois le spread — auquel s'ajoute le
glissement. Le mouvement, lui, croit a peu pres comme la racine de la duree.
Il existe donc un horizon en dessous duquel AUCUNE strategie ne paie, et
au-dessus duquel le peage devient supportable.

CE FICHIER NE SUPPOSE RIEN. Il mesure, sur les 879 860 barres M1 reelles :

  - le mouvement median et le mouvement des occasions FAVORABLES ;
  - ce qu'obtiendrait un selecteur PARFAIT, qui ne prendrait que les
    meilleures — c'est le plafond que le modele ne pourra pas depasser ;
  - ce qu'obtiendrait une entree AU HASARD, qui est le plancher a battre ;
  - et le nombre de trades par jour que chaque horizon permet.

LES DEUX OBJECTIFS SONT EN TENSION, et c'est tout l'enjeu : le nombre de
trades veut un horizon COURT, la rentabilite le veut LONG. Le tableau dit ou
se trouve le compromis.

    python mesure_horizon_m1.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import instruments as I

CACHE = "data_cache_XAUUSD_M1.pkl"
SYMBOLE = "XAUUSD"
# Les 5 % du haut, la meme selectivite que le systeme M5 utilisait.
Q = 0.05
HORIZONS = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240)


def main() -> int:
    p = I.INSTRUMENTS[SYMBOLE]
    cout = 2.0 * float(p["spread_bps"]) + 3.0      # aller-retour, en bps
    df = pd.read_pickle(CACHE)
    c = df["close"].to_numpy(np.float64)
    t = pd.to_datetime(df["time"])
    jours = (t.iloc[-1] - t.iloc[0]).total_seconds() / 86400.0
    print(f"\n{len(c):,} barres M1   {t.iloc[0]:%Y-%m-%d} -> {t.iloc[-1]:%Y-%m-%d}"
          f"   {jours:.0f} jours calendaires")
    print(f"cout d'un aller-retour : {cout:.2f} bps "
          f"({2*float(p['spread_bps']):.2f} de spread + 3.00 de glissement)")
    print("\nTous les rendements sont NETS du cout, en points de base.\n")

    print("  %-9s %9s %9s %10s %11s %10s"
          % ("horizon", "hasard", "sommet 5%", "plafond", "trades/jour",
             "gain/jour"))
    print("  " + "-" * 64)

    for h in HORIZONS:
        # Le rendement d'un LONG ouvert en t et ferme en t+h, net du cout.
        r = (c[h:] - c[:-h]) / c[:-h] * 1e4 - cout
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 1000:
            continue
        k = max(int(round(Q * n)), 50)

        # LE PLANCHER : une entree au hasard.
        hasard = float(np.mean(r))

        # LE PLAFOND : un selecteur parfait qui prend les k meilleures.
        # Aucun modele ne peut faire mieux sur cet horizon.
        plafond = float(np.mean(np.sort(r)[-k:]))

        # LE SOMMET D'UN TRI AU HASARD, pour situer le bruit : on tire un
        # score sans information et on prend ses 5 % du haut.
        rng = np.random.default_rng(0)
        som_bruit = float(np.mean(
            [np.mean(r[rng.choice(n, k, replace=False)]) for _ in range(30)]))

        # COMBIEN DE TRADES PAR JOUR. Une position occupe `h` barres ; en
        # n'en tenant qu'une a la fois, le marche en autorise au plus
        # `barres / h`. C'est une borne, pas une prevision.
        trades_jour = (n / h) / jours
        # Le gain quotidien d'un selecteur PARFAIT qui prendrait les 5 %
        # du haut, a une position a la fois.
        gain_jour = plafond * min(trades_jour, trades_jour * Q / Q) * Q

        print("  %-9s %+9.2f %+9.2f %+10.2f %11.1f %+10.2f"
              % (f"{h} min", hasard, som_bruit, plafond,
                 trades_jour, gain_jour))

    print("\n  hasard      : ce que rapporte une entree au hasard, nette")
    print("  sommet 5%%   : ce que rapporte un tri SANS information — le bruit")
    print("  plafond     : ce qu'obtiendrait un selecteur PARFAIT")
    print("  trades/jour : borne superieure, une position a la fois")
    print("  gain/jour   : plafond x nombre de trades, en bps de capital")
    print("\n  L'ECART entre `plafond` et `sommet 5%%` est ce qu'un modele")
    print("  peut esperer gagner. S'il est nul, l'horizon est inexploitable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
